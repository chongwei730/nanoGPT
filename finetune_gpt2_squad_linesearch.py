"""Fine-tune GPT-2 on SQuAD-like QA data but only use inputs (no answers) and optimize loss."""
import argparse
import math
import os
from functools import partial

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from lr_sched import LineSearchScheduler

from contextlib import nullcontext


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--pretrained_model', type=str, default='gpt2-medium')
    p.add_argument('--out_dir', type=str, default='/scratch.global/chen8596/out/gpt2-squad')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--gradient_accumulation_steps', type=int, default=8)
    p.add_argument('--block_size', type=int, default=1024)
    p.add_argument('--learning_rate', type=float, default=5e-5)
    p.add_argument('--weight_decay', type=float, default=0.0)
    p.add_argument('--max_iters', type=int, default=1000)
    p.add_argument('--eval_interval', type=int, default=200)
    p.add_argument('--eval_iters', type=int, default=50)
    p.add_argument('--save_interval', type=int, default=200)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--wandb_log', action='store_true')
    return p.parse_args()


def prepare_squad(tokenizer, block_size, split='train'):
    ds = load_dataset('squad', split=split)

    # Use only question + context as input; omit answer text (user requested no answers)
    texts = [f"Q: {ex['question'].strip()}\nContext: {ex['context'].strip()}\nA:" for ex in ds]

    tokenized = [tokenizer(t, truncation=True, max_length=block_size) for t in texts]
    input_ids = [x['input_ids'] for x in tokenized]
    return input_ids


def collate_fn(batch, pad_token_id):
    max_len = max(len(x) for x in batch)
    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, seq in enumerate(batch):
        input_ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        attention_mask[i, :len(seq)] = 1
    labels = input_ids.clone()
    labels[labels == pad_token_id] = -100
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}


def evaluate(model, dataloader, device, amp=False):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    ctx = torch.cuda.amp.autocast if amp else nullcontext
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            with ctx():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
            bsz = input_ids.size(0)
            total_loss += loss.item() * bsz
            total_tokens += bsz
    model.train()
    return total_loss / max(1, total_tokens)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.pretrained_model)
    model.resize_token_embeddings(len(tokenizer))
    device = torch.device(args.device)
    model.to(device)

    train_ids = prepare_squad(tokenizer, args.block_size, split='train')
    val_ids = prepare_squad(tokenizer, args.block_size, split='validation')

    train_loader = DataLoader(train_ids, batch_size=args.batch_size, shuffle=True,
                              collate_fn=partial(collate_fn, pad_token_id=tokenizer.pad_token_id))
    ls_train_loader = DataLoader(train_ids, batch_size=args.batch_size, shuffle=True,
                              collate_fn=partial(collate_fn, pad_token_id=tokenizer.pad_token_id))
    val_loader = DataLoader(val_ids, batch_size=args.batch_size, shuffle=False,
                            collate_fn=partial(collate_fn, pad_token_id=tokenizer.pad_token_id))

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-6)

    max_train_steps = args.max_iters
    num_warmup_steps = int(0.03 * max_train_steps)
    scheduler = LineSearchScheduler(optimizer=optimizer, 
                                model_paras=model.parameters(), 
                                num_search=16, start_lr=0, 
                                optimizer_type="AdamW", 
                                injection=True, 
                                search_mode="bisection", 
                                warmup_length=num_warmup_steps)
    linesearch_interval = 10000
    accum_steps = 32
    c1 = 0.01

    train_iter = iter(train_loader)
    ls_iter = iter(ls_train_loader)

    global_step = 0
    scaler = torch.cuda.amp.GradScaler()
    model.train()

    while global_step < max_train_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        if global_step % linesearch_interval == 0 or (
            global_step <= num_warmup_steps and global_step % num_warmup_steps == 0
        ):
            fixed_batches = []
            for _ in range(accum_steps):
                try:
                    ls_batch = next(ls_iter)
                except StopIteration:
                    ls_iter = iter(ls_train_loader)
                    ls_batch = next(ls_iter)

                x = ls_batch['input_ids'].to(device)
                m = ls_batch['attention_mask'].to(device)
                y = ls_batch['labels'].to(device)
                fixed_batches.append((x, m, y))

            def line_search_closure(require_grad=False):
                total_loss = torch.zeros((), device=device)
                for x_ls, m_ls, y_ls in fixed_batches:
                    outputs = model(input_ids=x_ls, attention_mask=m_ls, labels=y_ls)
                    loss = outputs.loss
                    total_loss += loss
                    if require_grad:
                        scaler.scale(loss / accum_steps).backward()
                return (total_loss / accum_steps).item()
        print("line search step")
        scheduler.step(
            line_search_closure,
            c1=c1,
            step=global_step,
            interval=linesearch_interval,
            condition="armijo",
            warmup_length=num_warmup_steps,
        )
        print("line search step over")

        with torch.cuda.amp.autocast():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

        loss_val = loss.item()
        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        global_step += 1

        if global_step % args.eval_interval == 0:
            val_loss = evaluate(model, val_loader, device)
            print(f"[step {global_step}] val loss: {val_loss:.4f}")

        if global_step % args.save_interval == 0:
            ckpt_dir = os.path.join(args.out_dir, f"ckpt-{global_step}")
            os.makedirs(ckpt_dir, exist_ok=True)
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            print(f"Saved checkpoint to {ckpt_dir}")


if __name__ == '__main__':
    main()

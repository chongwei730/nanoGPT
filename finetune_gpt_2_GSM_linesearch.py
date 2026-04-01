"""Fine-tune Hugging Face GPT-2 on GSM8K dataset.

This script follows the high-level logic of `train.py`: load model, prepare
data, training loop with gradient accumulation, periodic eval and checkpointing.

Usage examples:
python finetune_gpt_2_GSM.py --batch_size 2 --gradient_accumulation_steps 8 --max_iters 500
"""
import argparse
import math
import os
from functools import partial

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from lr_sched import LineSearchScheduler

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--pretrained_model', type=str, default='gpt2-medium')
    p.add_argument('--out_dir', type=str, default='/scratch.global/chen8596/out/gpt2-gsm8k')
    p.add_argument('--batch_size', type=int, default=2)
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
    p.add_argument('--wandb_log', action='store_true', help='enable logging to Weights & Biases')
    p.add_argument('--wandb_project', type=str, default='gpt2-gsm8k')
    p.add_argument('--wandb_run_name', type=str, default="fine_tune_cosine")
    return p.parse_args()


def prepare_gsm8k(tokenizer, block_size, split='train'):
    ds = load_dataset('gsm8k', 'main', split=split)

    # Format: Prompt: Q: {question}\nA:
    def format_example(ex):
        q = ex['question'].strip()
        a = ex.get('answer', '').strip()
        text = f"Q: {q}\nA: {a}\n" if a else f"Q: {q}\nA:"
        return text

    texts = [format_example(x) for x in ds]

    def tokenize_fn(examples):
        return tokenizer(examples, truncation=True, max_length=block_size)

    tokenized = list(map(lambda t: tokenizer(t, truncation=True, max_length=block_size), texts))

    # convert to list of input_ids
    input_ids = [x['input_ids'] for x in tokenized]
    return input_ids


def collate_fn(batch, pad_token_id):
    # batch: list of lists of input_ids
    max_len = max(len(x) for x in batch)
    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, seq in enumerate(batch):
        input_ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        attention_mask[i, :len(seq)] = 1
    labels = input_ids.clone()
    labels[labels == pad_token_id] = -100
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
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


    # optional wandb init
    if args.wandb_log:
        import wandb
        run_name = args.wandb_run_name if args.wandb_run_name is not None else None
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    # Prepare datasets
    train_ids = prepare_gsm8k(tokenizer, args.block_size, split='train')
    try:
        val_ids = prepare_gsm8k(tokenizer, args.block_size, split='validation')
    except Exception:
        # some dataset versions use 'test'
        val_ids = prepare_gsm8k(tokenizer, args.block_size, split='test')

    train_loader = DataLoader(train_ids, batch_size=args.batch_size, shuffle=True,
                              collate_fn=partial(collate_fn, pad_token_id=tokenizer.pad_token_id))
    ls_train_loader = DataLoader(train_ids, batch_size=args.batch_size, shuffle=True,
                              collate_fn=partial(collate_fn, pad_token_id=tokenizer.pad_token_id))
    val_loader = DataLoader(val_ids, batch_size=args.batch_size, shuffle=False,
                            collate_fn=partial(collate_fn, pad_token_id=tokenizer.pad_token_id))

    # optimizer + scheduler
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-6)

    num_update_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    max_train_steps = args.max_iters
    # num_warmup_steps = int(0.03 * max_train_steps)
    num_warmup_steps = 1000
    scheduler = LineSearchScheduler(optimizer=optimizer, 
                                model_paras=model.parameters(), 
                                num_search=16, start_lr=0, 
                                optimizer_type="AdamW", 
                                injection=True, 
                                search_mode="bisection", 
                                warmup_length=num_warmup_steps)
    linesearch_interval = 1000
    accum_steps = 32
    c1 = 0.03
    ls_iter = iter(ls_train_loader)

    # iterators
    train_iter = iter(train_loader)
    ls_iter = iter(ls_train_loader)

    # training state
    global_step = 0
    # AMP removed: use standard FP training
    scaler = None
    model.train()
    line_search_closure = None

    while global_step < max_train_steps:

        # ===== 1. fetch batch =====
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # ===== 2. line search =====
       
        if ((global_step) % linesearch_interval == 0) or (
            global_step <= num_warmup_steps and ((global_step) % num_warmup_steps == 0)
        ):
            print(f"[step {global_step}] LINESEARCH")

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
                if require_grad:
                    optimizer.zero_grad()
                total_loss = torch.zeros((), device=device)

                for x_ls, m_ls, y_ls in fixed_batches:
                    outputs = model(
                        input_ids=x_ls,
                        attention_mask=m_ls,
                        labels=y_ls,
                    )
                    loss = outputs.loss
                    total_loss += loss

                    if require_grad:
                        (loss / accum_steps).backward()

                # If gradients were produced for the closure, clip them
                if require_grad:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                return (total_loss / accum_steps).item()

        # ===== 3. scheduler =====
        scheduler.step(
            line_search_closure,
            c1=c1,
            step=global_step,
            interval=linesearch_interval,
            condition="armijo",
            warmup_length=num_warmup_steps,
        )

        # ===== 4. forward + backward =====
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss

        loss_val = loss.item()
        loss.backward()

        # ===== 5. optimizer step =====
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        global_step += 1

        # ===== 6. logging =====
        if args.wandb_log:
            import wandb
            lr = (
                scheduler.get_last_lr()[0]
                if hasattr(scheduler, "get_last_lr")
                else optimizer.param_groups[0]["lr"]
            )
            wandb.log(
                {"train/loss": loss_val, "train/lr": lr},
                step=global_step,
            )

        # ===== 7. evaluation =====
        if global_step % args.eval_interval == 0:
            val_loss = evaluate(model, val_loader, device)
            print(f"[step {global_step}] val loss: {val_loss:.4f}")

            if args.wandb_log:
                wandb.log({"val/loss": val_loss}, step=global_step)

        # ===== 8. checkpoint =====
        if global_step % args.save_interval == 0:
            ckpt_dir = os.path.join(args.out_dir, f"ckpt-{global_step}")
            os.makedirs(ckpt_dir, exist_ok=True)

            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)

            print(f"Saved checkpoint to {ckpt_dir}")


if __name__ == '__main__':
    main()

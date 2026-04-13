# config for training GPT-2 355M on OpenWebText for about 7B tokens on 4 GPUs

wandb_run_name = 'gpt2-355M-7B'

# 36 * 14 * 1024 = 516,096 tokens / iter
batch_size = 14
block_size = 1024
gradient_accumulation_steps = 36

# about 7.02B tokens total
max_iters = 13600
lr_decay_iters = 13600
warmup_iters = 100

# model size
n_layer = 24
n_head = 16
n_embd = 1024

# eval / logging
eval_interval = 100
eval_iters = 200
log_interval = 10

weight_decay = 1e-1

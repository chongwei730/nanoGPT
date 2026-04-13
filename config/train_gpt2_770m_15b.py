# config for training GPT-2 770M on OpenWebText for about 15B tokens on 4 GPUs

wandb_run_name = 'gpt2-770M-15B'

# 60 * 8 * 1024 = 491,520 tokens / iter
batch_size = 8
block_size = 1024
gradient_accumulation_steps = 60

# about 15.00B tokens total
max_iters = 30518
lr_decay_iters = 30518
warmup_iters = 300

# model size
n_layer = 36
n_head = 20
n_embd = 1280

# eval / logging
eval_interval = 100
eval_iters = 200
log_interval = 10

weight_decay = 1e-1

batch_size = 8
block_size = 1024
max_length = 1024
gradient_accumulation_steps = 4 * 4

max_iters = 5000
lr_decay_iters = 5000
warmup_iters = 100

eval_interval = 100
eval_iters = 50
log_interval = 10

weight_decay = 1e-1
compile = True
dataset = "c4"
dataset_config_name = "en"
tokenizer_name = "t5-base"
llama_config_path = "llama_config/llama_350m.json"
dataloader_num_workers = 4
preprocessing_num_workers = 8
tokenizer_batch_size = 1000
tokenized_data_cache_dir = "/scratch.global/chen8596/tokenized_dataset_c4/c4_en_t5-base_len1024"
hf_data_cache_dir = "/scratch.global/chen8596/hf_datasets_cache"
overwrite_tokenized_cache = False

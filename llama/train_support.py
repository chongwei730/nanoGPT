import itertools
import json
import os
import pickle
import re
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

from datasets import load_dataset, load_from_disk
from schedulefree_import import load_adamw_schedulefree
from transformers import AutoTokenizer
from transformers.models.llama.configuration_llama import LlamaConfig

from llama.dataloader import PreprocessedIterableDataset
from llama.modeling_llama import LlamaForCausalLM

try:
    from datasets.distributed import split_dataset_by_node
except ImportError:  # pragma: no cover - compatibility fallback
    split_dataset_by_node = None


def load_llama_config(llama_config_path: str, vocab_size: int, max_length: int) -> Tuple[LlamaConfig, Dict]:
    with open(llama_config_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)

    if "max_position_embeddings" not in config_data:
        config_data["max_position_embeddings"] = int(config_data.get("max_sequence_length", max_length))
    config_data["max_position_embeddings"] = int(max_length)
    config_data["max_sequence_length"] = int(max_length)
    config_data["vocab_size"] = int(vocab_size)
    if "num_key_value_heads" not in config_data and "num_attention_heads" in config_data:
        config_data["num_key_value_heads"] = int(config_data["num_attention_heads"])
    config_data["use_cache"] = False

    return LlamaConfig(**config_data), config_data


def build_tokenizer(tokenizer_name: str, max_length: int):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, model_max_length=max_length, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not tokenizer.is_fast:
        raise ValueError(f"Tokenizer {tokenizer_name!r} is not a fast tokenizer.")
    return tokenizer


def sanitize_path_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def default_tokenized_cache_dir(
    dataset_name: str,
    dataset_config_name: str,
    tokenizer_name: str,
    max_length: int,
):
    dataset_part = sanitize_path_component(dataset_name)
    config_part = sanitize_path_component(dataset_config_name)
    tokenizer_part = sanitize_path_component(tokenizer_name)
    return os.path.join(
        "/scratch.global/chen8596",
        "tokenized_dataset_c4",
        f"{dataset_part}_{config_part}_{tokenizer_part}_len{int(max_length)}",
    )


def choose_token_dtype(vocab_size: int):
    if vocab_size <= np.iinfo(np.uint16).max:
        return np.uint16
    if vocab_size <= np.iinfo(np.uint32).max:
        return np.uint32
    raise ValueError(f"Unsupported vocab size for compact token cache: {vocab_size}")


def resolve_map_dataset(dataset_name: str, dataset_config_name: str, split: str, cache_dir: str = ""):
    candidates = [(dataset_name, dataset_config_name)]
    if dataset_name == "c4":
        candidates.extend(
            [
                ("allenai/c4", dataset_config_name),
                ("allenai/c4", "default"),
            ]
        )
    elif dataset_name == "allenai/c4":
        candidates.append(("c4", dataset_config_name))

    errors = []
    seen = set()
    for candidate_name, candidate_config in candidates:
        key = (candidate_name, candidate_config)
        if key in seen:
            continue
        seen.add(key)
        try:
            kwargs = {"split": split}
            if cache_dir:
                kwargs["cache_dir"] = cache_dir
            return load_dataset(candidate_name, candidate_config, **kwargs)
        except Exception as exc:  # pragma: no cover - runtime fallback
            errors.append(f"{candidate_name}/{candidate_config}: {type(exc).__name__}: {exc}")

    joined_errors = "\n".join(errors)
    raise RuntimeError(
        "Failed to load map-style dataset with all known C4-compatible sources.\n"
        f"Tried:\n{joined_errors}"
    )


def resolve_streaming_dataset(dataset_name: str, dataset_config_name: str, split: str):
    candidates = [(dataset_name, dataset_config_name)]
    if dataset_name == "c4":
        candidates.extend(
            [
                ("allenai/c4", dataset_config_name),
                ("allenai/c4", "default"),
            ]
        )

    errors = []
    seen = set()
    for candidate_name, candidate_config in candidates:
        key = (candidate_name, candidate_config)
        if key in seen:
            continue
        seen.add(key)
        try:
            return load_dataset(candidate_name, candidate_config, split=split, streaming=True)
        except Exception as exc:  # pragma: no cover - runtime fallback
            errors.append(f"{candidate_name}/{candidate_config}: {type(exc).__name__}: {exc}")

    joined_errors = "\n".join(errors)
    raise RuntimeError(
        "Failed to load streaming dataset with all known C4-compatible sources.\n"
        f"Tried:\n{joined_errors}"
    )


def tokenize_and_group_text_dataset(raw_dataset, tokenizer, block_size: int, batch_size: int, num_proc: int):
    column_names = list(raw_dataset.column_names)
    text_column = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(
            examples[text_column],
            add_special_tokens=True,
            return_attention_mask=False,
            truncation=False,
        )

    tokenized = raw_dataset.map(
        tokenize_function,
        batched=True,
        batch_size=batch_size,
        remove_columns=column_names,
        num_proc=num_proc,
        desc="Tokenizing text with fast tokenizer",
    )

    def group_texts(examples):
        concatenated = []
        for ids in examples["input_ids"]:
            concatenated.extend(ids)
        total_length = (len(concatenated) // block_size) * block_size
        if total_length == 0:
            return {"input_ids": []}
        return {
            "input_ids": [
                concatenated[index : index + block_size]
                for index in range(0, total_length, block_size)
            ]
        }

    grouped = tokenized.map(
        group_texts,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        desc=f"Packing token stream into {block_size}-token blocks",
    )
    grouped.set_format(type="torch", columns=["input_ids"])
    return grouped


def load_or_prepare_tokenized_dataset(
    dataset_name: str,
    dataset_config_name: str,
    split: str,
    tokenizer,
    block_size: int,
    tokenized_data_cache_dir: str,
    hf_data_cache_dir: str,
    preprocessing_num_workers: int,
    tokenizer_batch_size: int,
    overwrite_tokenized_cache: bool,
):
    split_cache_dir = os.path.join(tokenized_data_cache_dir, sanitize_path_component(split))
    if os.path.isdir(split_cache_dir) and not overwrite_tokenized_cache:
        dataset = load_from_disk(split_cache_dir)
        dataset.set_format(type="torch", columns=["input_ids"])
        return dataset

    os.makedirs(tokenized_data_cache_dir, exist_ok=True)
    raw_dataset = resolve_map_dataset(
        dataset_name=dataset_name,
        dataset_config_name=dataset_config_name,
        split=split,
        cache_dir=hf_data_cache_dir,
    )
    tokenized_dataset = tokenize_and_group_text_dataset(
        raw_dataset=raw_dataset,
        tokenizer=tokenizer,
        block_size=block_size,
        batch_size=tokenizer_batch_size,
        num_proc=preprocessing_num_workers,
    )
    tokenized_dataset.save_to_disk(split_cache_dir)
    tokenized_dataset.set_format(type="torch", columns=["input_ids"])
    return tokenized_dataset


def build_cached_lm_dataloader(
    dataset,
    batch_size: int,
    global_rank: int,
    world_size: int,
    num_workers: int,
    shuffle: bool,
):
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=global_rank,
            shuffle=shuffle,
            drop_last=False,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=bool(shuffle and sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=bool(num_workers > 0),
    ), sampler


def build_c4_dataloader(
    dataset_name: str,
    dataset_config_name: str,
    split: str,
    tokenizer,
    batch_size: int,
    max_length: int,
    seed: int,
    shuffle_buffer_size: int,
    global_rank: int,
    world_size: int,
    num_workers: int,
):
    data = resolve_streaming_dataset(dataset_name=dataset_name, dataset_config_name=dataset_config_name, split=split)
    if shuffle_buffer_size > 0:
        data = data.shuffle(buffer_size=shuffle_buffer_size, seed=seed)
    if world_size > 1:
        if split_dataset_by_node is not None:
            data = split_dataset_by_node(data, rank=global_rank, world_size=world_size)
        else:
            data = data.shard(num_shards=world_size, index=global_rank)
    dataset = PreprocessedIterableDataset(
        data=data,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
    )
    return DataLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        persistent_workers=bool(num_workers > 0),
    )


class StreamingBatchIterator:
    def __init__(self, loader_factory):
        self.loader_factory = loader_factory
        self.loader = None
        self.iterator = None
        self.reset()

    def reset(self):
        self.loader = self.loader_factory()
        self.iterator = iter(self.loader)

    def next_batch(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.reset()
            return next(self.iterator)


class DataLoaderBatchIterator:
    def __init__(self, loader_factory):
        self.loader_factory = loader_factory
        self.loader = None
        self.sampler = None
        self.iterator = None
        self.epoch = 0
        self.reset()

    def reset(self):
        self.loader, self.sampler = self.loader_factory()
        if self.sampler is not None:
            self.sampler.set_epoch(self.epoch)
        self.epoch += 1
        self.iterator = iter(self.loader)

    def next_batch(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.reset()
            return next(self.iterator)


def load_token_data(cache_dir: str, split: str, data_backend: str):
    meta_path = os.path.join(cache_dir, "meta.pkl")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing token cache metadata: {meta_path}")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    dtype_name = meta["token_dtype"]
    dtype = getattr(np, dtype_name)
    path = os.path.join(cache_dir, f"{split}.bin")
    if data_backend == "memmap":
        return np.memmap(path, dtype=dtype, mode="r"), meta
    if data_backend == "ram":
        return np.fromfile(path, dtype=dtype), meta
    raise ValueError(f"Unsupported data_backend: {data_backend}")


def _write_text_stream_to_bin(
    dataset_name: str,
    dataset_config_name: str,
    split: str,
    tokenizer,
    output_path: str,
    token_dtype,
    text_batch_size: int,
):
    data = resolve_streaming_dataset(dataset_name=dataset_name, dataset_config_name=dataset_config_name, split=split)
    token_count = 0
    batch_texts = []
    with open(output_path, "wb") as f:
        for example in data:
            batch_texts.append(example["text"])
            if len(batch_texts) < text_batch_size:
                continue
            encoded = tokenizer(
                batch_texts,
                add_special_tokens=True,
                return_attention_mask=False,
                truncation=False,
            )["input_ids"]
            flat_tokens = np.fromiter(itertools.chain.from_iterable(encoded), dtype=token_dtype)
            flat_tokens.tofile(f)
            token_count += int(flat_tokens.size)
            batch_texts = []
        if batch_texts:
            encoded = tokenizer(
                batch_texts,
                add_special_tokens=True,
                return_attention_mask=False,
                truncation=False,
            )["input_ids"]
            flat_tokens = np.fromiter(itertools.chain.from_iterable(encoded), dtype=token_dtype)
            flat_tokens.tofile(f)
            token_count += int(flat_tokens.size)
    return token_count


def _write_text_stream_shard_to_bin(
    dataset_name: str,
    dataset_config_name: str,
    split: str,
    tokenizer_name: str,
    max_length: int,
    output_path: str,
    token_dtype_name: str,
    text_batch_size: int,
    num_shards: int,
    shard_index: int,
):
    tokenizer = build_tokenizer(tokenizer_name=tokenizer_name, max_length=max_length)
    token_dtype = getattr(np, token_dtype_name)
    data = resolve_streaming_dataset(dataset_name=dataset_name, dataset_config_name=dataset_config_name, split=split)
    if num_shards > 1:
        data = data.shard(num_shards=num_shards, index=shard_index)

    token_count = 0
    batch_texts = []
    with open(output_path, "wb") as f:
        for example in data:
            batch_texts.append(example["text"])
            if len(batch_texts) < text_batch_size:
                continue
            encoded = tokenizer(
                batch_texts,
                add_special_tokens=True,
                return_attention_mask=False,
                truncation=False,
            )["input_ids"]
            flat_tokens = np.fromiter(itertools.chain.from_iterable(encoded), dtype=token_dtype)
            flat_tokens.tofile(f)
            token_count += int(flat_tokens.size)
            batch_texts = []
        if batch_texts:
            encoded = tokenizer(
                batch_texts,
                add_special_tokens=True,
                return_attention_mask=False,
                truncation=False,
            )["input_ids"]
            flat_tokens = np.fromiter(itertools.chain.from_iterable(encoded), dtype=token_dtype)
            flat_tokens.tofile(f)
            token_count += int(flat_tokens.size)
    return token_count


def _merge_bin_parts(part_paths, output_path: str):
    with open(output_path, "wb") as output_file:
        for part_path in part_paths:
            with open(part_path, "rb") as input_file:
                shutil.copyfileobj(input_file, output_file, length=16 * 1024 * 1024)


def _write_text_stream_to_bin_parallel(
    dataset_name: str,
    dataset_config_name: str,
    split: str,
    tokenizer,
    output_path: str,
    token_dtype,
    text_batch_size: int,
    num_shards: int,
):
    if num_shards <= 1:
        return _write_text_stream_to_bin(
            dataset_name=dataset_name,
            dataset_config_name=dataset_config_name,
            split=split,
            tokenizer=tokenizer,
            output_path=output_path,
            token_dtype=token_dtype,
            text_batch_size=text_batch_size,
        )

    work_dir = f"{output_path}.parts"
    os.makedirs(work_dir, exist_ok=True)
    part_paths = [os.path.join(work_dir, f"part-{shard_index:05d}.bin") for shard_index in range(num_shards)]
    futures = []
    token_counts = []
    try:
        with ProcessPoolExecutor(max_workers=num_shards) as executor:
            for shard_index, part_path in enumerate(part_paths):
                futures.append(
                    executor.submit(
                        _write_text_stream_shard_to_bin,
                        dataset_name,
                        dataset_config_name,
                        split,
                        tokenizer.name_or_path,
                        tokenizer.model_max_length,
                        part_path,
                        np.dtype(token_dtype).name,
                        text_batch_size,
                        num_shards,
                        shard_index,
                    )
                )
            for shard_index, future in enumerate(futures):
                token_count = future.result()
                token_counts.append(token_count)
                print(
                    f"finished tokenizing split={split} shard={shard_index + 1}/{num_shards} "
                    f"tokens={token_count:,}"
                )

        _merge_bin_parts(part_paths=part_paths, output_path=output_path)
        return int(sum(token_counts))
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def prepare_token_bin_cache(
    dataset_name: str,
    dataset_config_name: str,
    tokenizer,
    cache_dir: str,
    overwrite_tokenized_cache: bool,
    text_batch_size: int,
    num_proc: int = 1,
):
    os.makedirs(cache_dir, exist_ok=True)
    meta_path = os.path.join(cache_dir, "meta.pkl")
    train_path = os.path.join(cache_dir, "train.bin")
    val_path = os.path.join(cache_dir, "val.bin")

    if (
        not overwrite_tokenized_cache
        and os.path.exists(meta_path)
        and os.path.exists(train_path)
        and os.path.exists(val_path)
    ):
        with open(meta_path, "rb") as f:
            return pickle.load(f)

    token_dtype = choose_token_dtype(tokenizer.vocab_size)
    train_tokens = _write_text_stream_to_bin_parallel(
        dataset_name=dataset_name,
        dataset_config_name=dataset_config_name,
        split="train",
        tokenizer=tokenizer,
        output_path=train_path,
        token_dtype=token_dtype,
        text_batch_size=text_batch_size,
        num_shards=max(1, int(num_proc)),
    )
    val_tokens = _write_text_stream_to_bin_parallel(
        dataset_name=dataset_name,
        dataset_config_name=dataset_config_name,
        split="validation",
        tokenizer=tokenizer,
        output_path=val_path,
        token_dtype=token_dtype,
        text_batch_size=text_batch_size,
        num_shards=max(1, int(num_proc)),
    )

    meta = {
        "vocab_size": int(tokenizer.vocab_size),
        "token_dtype": np.dtype(token_dtype).name,
        "tokenizer_name": tokenizer.name_or_path,
        "train_tokens": int(train_tokens),
        "val_tokens": int(val_tokens),
    }
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)
    return meta


def token_bin_cache_is_ready(cache_dir: str) -> bool:
    meta_path = os.path.join(cache_dir, "meta.pkl")
    train_path = os.path.join(cache_dir, "train.bin")
    val_path = os.path.join(cache_dir, "val.bin")
    return (
        os.path.exists(meta_path)
        and os.path.exists(train_path)
        and os.path.exists(val_path)
    )


def wait_for_token_bin_cache(cache_dir: str, poll_interval_seconds: float = 5.0):
    while not token_bin_cache_is_ready(cache_dir):
        time.sleep(poll_interval_seconds)


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: str, device_type: str) -> Dict[str, torch.Tensor]:
    moved = {}
    for key, value in batch.items():
        if device_type == "cuda":
            moved[key] = value.pin_memory().to(device, non_blocking=True)
        else:
            moved[key] = value.to(device)
    return moved


def prepare_batch(batch: Dict[str, torch.Tensor], device: str, device_type: str, to_device: bool = True):
    if to_device:
        batch = move_batch_to_device(batch, device=device, device_type=device_type)
    input_ids = batch["input_ids"]
    attention_mask = batch.get("attention_mask")
    labels = input_ids.clone()
    if attention_mask is not None:
        labels = labels.masked_fill(attention_mask == 0, -100)
    return input_ids, attention_mask, labels


def strip_unwanted_prefix(state_dict, unwanted_prefix="_orig_mod."):
    for key in list(state_dict.keys()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix):]] = state_dict.pop(key)
    return state_dict


def build_llama_model(llama_config_path: str, tokenizer_vocab_size: int, max_length: int):
    config, config_dict = load_llama_config(
        llama_config_path=llama_config_path,
        vocab_size=tokenizer_vocab_size,
        max_length=max_length,
    )
    model = LlamaForCausalLM(config)
    return model, config_dict


def build_optimizer(
    model,
    learning_rate: float,
    weight_decay: float,
    betas,
    device_type: str,
    optimizer_type: str = "AdamW",
):
    decay_params = []
    no_decay_params = []
    seen = set()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in seen:
            continue
        seen.add(id(param))
        lowered = name.lower()
        if (
            param.ndim < 2
            or lowered.endswith("bias")
            or "norm" in lowered
            or "embed_tokens" in lowered
            or "lm_head" in lowered
        ):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay, "lr": learning_rate},
        {"params": no_decay_params, "weight_decay": 0.0, "lr": learning_rate},
    ]

    if optimizer_type == "AdamWScheduleFree":
        AdamWScheduleFree = load_adamw_schedulefree()
        print("using optimizer: AdamWScheduleFree")
        return AdamWScheduleFree(
            param_groups,
            lr=learning_rate,
            betas=betas,
            weight_decay=weight_decay,
        )

    if optimizer_type != "AdamW":
        raise ValueError(f"Unsupported optimizer_type: {optimizer_type}")

    fused_available = device_type == "cuda" and "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
    extra_kwargs = {"fused": True} if fused_available else {}
    print(f"using optimizer: {optimizer_type}, fused={bool(extra_kwargs)}")
    return torch.optim.AdamW(param_groups, betas=betas, **extra_kwargs)

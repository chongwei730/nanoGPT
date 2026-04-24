#!/usr/bin/env python3
"""
Download a C4 split to scratch and tokenize it with the T5 tokenizer.
"""

import argparse
import json
import os
import shutil

from datasets import load_dataset, load_from_disk

from llama.train_support import (
    build_tokenizer,
    sanitize_path_component,
    tokenize_and_group_text_dataset,
)


def default_raw_dataset_dir(dataset_name: str, dataset_config_name: str, split: str) -> str:
    return os.path.join(
        "/scratch.global/chen8596",
        "raw_datasets",
        f"{sanitize_path_component(dataset_name)}_{sanitize_path_component(dataset_config_name)}_{sanitize_path_component(split)}",
    )


def default_tokenized_dataset_dir(
    dataset_name: str,
    dataset_config_name: str,
    split: str,
    tokenizer_name: str,
    block_size: int,
) -> str:
    return os.path.join(
        "/scratch.global/chen8596",
        "tokenized_dataset_c4",
        (
            f"{sanitize_path_component(dataset_name)}_"
            f"{sanitize_path_component(dataset_config_name)}_"
            f"{sanitize_path_component(split)}_"
            f"{sanitize_path_component(tokenizer_name)}_"
            f"len{int(block_size)}"
        ),
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download a C4 split to scratch and tokenize it with T5."
    )
    parser.add_argument("--dataset", default="c4")
    parser.add_argument("--dataset-config-name", default="en")
    parser.add_argument("--split", default="train")
    parser.add_argument("--tokenizer-name", default="t5-base")
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument(
        "--hf-cache-dir",
        default="/scratch.global/chen8596/hf_cache",
    )
    parser.add_argument("--raw-save-dir", default="")
    parser.add_argument("--tokenized-save-dir", default="")
    parser.add_argument("--tokenizer-batch-size", type=int, default=1000)
    parser.add_argument("--preprocessing-num-workers", type=int, default=8)
    parser.add_argument("--overwrite-raw", action="store_true")
    parser.add_argument("--overwrite-tokenized", action="store_true")
    return parser.parse_args()


def ensure_parent_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def load_or_download_raw_dataset(args, raw_save_dir: str):
    if os.path.isdir(raw_save_dir) and not args.overwrite_raw:
        return load_from_disk(raw_save_dir)

    ensure_parent_dir(raw_save_dir)
    if os.path.isdir(raw_save_dir):
        shutil.rmtree(raw_save_dir)
    raw_dataset = load_dataset(
        args.dataset,
        args.dataset_config_name,
        split=args.split,
        cache_dir=args.hf_cache_dir,
    )
    raw_dataset.save_to_disk(raw_save_dir)
    return raw_dataset


def main():
    args = parse_args()
    if args.max_length != args.block_size:
        raise ValueError("max_length and block_size must match.")

    raw_save_dir = args.raw_save_dir or default_raw_dataset_dir(
        dataset_name=args.dataset,
        dataset_config_name=args.dataset_config_name,
        split=args.split,
    )
    tokenized_save_dir = args.tokenized_save_dir or default_tokenized_dataset_dir(
        dataset_name=args.dataset,
        dataset_config_name=args.dataset_config_name,
        split=args.split,
        tokenizer_name=args.tokenizer_name,
        block_size=args.block_size,
    )

    raw_dataset = load_or_download_raw_dataset(args=args, raw_save_dir=raw_save_dir)

    if os.path.isdir(tokenized_save_dir) and not args.overwrite_tokenized:
        tokenized_dataset = load_from_disk(tokenized_save_dir)
    else:
        tokenizer = build_tokenizer(
            tokenizer_name=args.tokenizer_name,
            max_length=args.max_length,
        )
        ensure_parent_dir(tokenized_save_dir)
        if os.path.isdir(tokenized_save_dir):
            shutil.rmtree(tokenized_save_dir)
        tokenized_dataset = tokenize_and_group_text_dataset(
            raw_dataset=raw_dataset,
            tokenizer=tokenizer,
            block_size=args.block_size,
            batch_size=args.tokenizer_batch_size,
            num_proc=args.preprocessing_num_workers,
        )
        tokenized_dataset.save_to_disk(tokenized_save_dir)

    metadata = {
        "dataset": args.dataset,
        "dataset_config_name": args.dataset_config_name,
        "split": args.split,
        "tokenizer_name": args.tokenizer_name,
        "block_size": int(args.block_size),
        "max_length": int(args.max_length),
        "hf_cache_dir": args.hf_cache_dir,
        "raw_save_dir": raw_save_dir,
        "tokenized_save_dir": tokenized_save_dir,
        "raw_num_rows": int(len(raw_dataset)),
        "tokenized_num_rows": int(len(tokenized_dataset)),
    }

    metadata_path = os.path.join(tokenized_save_dir, "prepare_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

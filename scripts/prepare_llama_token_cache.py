#!/usr/bin/env python3
"""
Prepare a complete token cache for LLaMA experiments.
"""

import argparse
import os
import pickle

from llama.train_support import build_tokenizer, prepare_token_bin_cache


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare tokenized train/val bins for LLaMA runs.")
    parser.add_argument("--dataset", default="c4")
    parser.add_argument("--dataset-config-name", default="en")
    parser.add_argument("--tokenizer-name", default="t5-base")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument(
        "--cache-dir",
        default="/scratch.global/chen8596/tokenized_dataset/c4_en_t5-base_len1024",
    )
    parser.add_argument("--text-batch-size", type=int, default=1000)
    parser.add_argument("--num-proc", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    tokenizer = build_tokenizer(tokenizer_name=args.tokenizer_name, max_length=args.max_length)
    meta = prepare_token_bin_cache(
        dataset_name=args.dataset,
        dataset_config_name=args.dataset_config_name,
        tokenizer=tokenizer,
        cache_dir=args.cache_dir,
        overwrite_tokenized_cache=args.overwrite,
        text_batch_size=args.text_batch_size,
        num_proc=args.num_proc,
    )

    train_path = os.path.join(args.cache_dir, "train.bin")
    val_path = os.path.join(args.cache_dir, "val.bin")
    meta_path = os.path.join(args.cache_dir, "meta.pkl")

    if not (os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(meta_path)):
        raise RuntimeError(f"Incomplete token cache in {args.cache_dir}")

    with open(meta_path, "rb") as f:
        saved_meta = pickle.load(f)

    print(f"cache_dir={args.cache_dir}")
    print(f"train.bin bytes={os.path.getsize(train_path)}")
    print(f"val.bin bytes={os.path.getsize(val_path)}")
    print(f"meta={saved_meta}")
    print(f"prepare_token_bin_cache_return={meta}")


if __name__ == "__main__":
    main()

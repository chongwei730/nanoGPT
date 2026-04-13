import os


def resolve_resume_checkpoint(out_dir):
    candidates = [
        os.path.join(out_dir, "ckpt.pt"),
        os.path.join(out_dir, "ckpt_last.pt"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"No checkpoint found in {out_dir!r}. Tried: "
        + ", ".join(repr(path) for path in candidates)
    )

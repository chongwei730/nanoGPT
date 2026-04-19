import sys
from importlib import import_module
from pathlib import Path


def load_adamw_schedulefree():
    try:
        return import_module("schedulefree").AdamWScheduleFree
    except ImportError as original_exc:
        vendored_root = Path(__file__).resolve().parent / "schedule_free"
        if vendored_root.is_dir():
            vendored_root_str = str(vendored_root)
            if vendored_root_str not in sys.path:
                sys.path.insert(0, vendored_root_str)
            try:
                return import_module("schedulefree").AdamWScheduleFree
            except ImportError:
                pass

        raise ImportError(
            "optimizer_type=AdamWScheduleFree requires the `schedulefree` package. "
            "Install it in the runtime environment or keep the vendored source tree "
            f"available at {vendored_root}."
        ) from original_exc

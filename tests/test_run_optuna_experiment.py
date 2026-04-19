import json
import os
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import yaml

import run_optuna_experiment


class RunOptunaExperimentIntegrationTest(unittest.TestCase):
    def test_tiny_gpu_/work/nvme/bgop/cchen47/experiment_runs_end_to_end(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for the real torchrun integration path.")

        repo_root = Path(__file__).resolve().parents[1]
        output_root = repo_root / "/work/nvme/bgop/cchen47/experiment_runs"

        with tempfile.TemporaryDirectory(dir="/tmp") as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            dataset_dir = tmpdir / "tiny_openwebtext"
            dataset_dir.mkdir()

            vocab_size = 32
            train_tokens = (np.arange(256, dtype=np.uint16) % vocab_size).astype(np.uint16)
            val_tokens = (np.arange(128, dtype=np.uint16) % vocab_size).astype(np.uint16)
            train_tokens.tofile(dataset_dir / "train.bin")
            val_tokens.tofile(dataset_dir / "val.bin")
            with open(dataset_dir / "meta.pkl", "wb") as f:
                pickle.dump({"vocab_size": vocab_size}, f)

            table_path = tmpdir / "table.txt"
            table_path.write_text(
                "\n".join(
                    [
                        "| family | dataset | model_size | tokens_b | train_target | test_target | runtime_hours | notes |",
                        "| --- | --- | --- | --- | --- | --- | --- | --- |",
                        "| TinyGPT | TinyOpenWebText | tiny | 0.0 | 0.0 | 0.0 | 1 | integration test |",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            dataset_override = os.path.relpath(
                dataset_dir,
                "/work/nvme/bgop/cchen47/experiment_runs/nanogpt_data",
            )
            config_path = tmpdir / "tiny_optuna.yaml"
            config = {
                "experiment": {
                    "name": "tiny_optuna_gpu",
                    "train_script": "train.py",
                    "output_root": str(output_root),
                    "target_family": "TinyGPT",
                    "target_dataset": "TinyOpenWebText",
                    "target_model_size": "tiny",
                },
                "hyperparameters": {
                    "learning_rate": {
                        "type": "log_uniform",
                        "range": [1e-4, 3e-4],
                    }
                },
                "task": {
                    "train_metric": "train_loss",
                    "test_metric": "val_loss",
                    "metric_mode": "min",
                    "max_running_time_per_trial_hours": 1.0,
                },
                "fixed_args": {
                    "dataset": dataset_override,
                    "compile": False,
                    "dtype": "float32",
                    "gradient_accumulation_steps": 1,
                    "batch_size": 2,
                    "block_size": 8,
                    "n_layer": 1,
                    "n_head": 1,
                    "n_embd": 16,
                    "dropout": 0.0,
                    "bias": False,
                    "weight_decay": 0.0,
                    "eval_interval": 1,
                    "log_interval": 1,
                    "eval_iters": 1,
                    "max_iters": 1,
                    "warmup_iters": 0,
                    "lr_decay_iters": 1,
                    "min_lr": 1e-4,
                    "always_save_checkpoint": False,
                },
                "checkpoint": {
                    "save_last": True,
                },
                "launch": {
                    "mode": "torchrun",
                    "nproc_per_node": 1,
                },
                "optuna": {
                    "max_study_time_hours": 1.0,
                    "pruning": {
                        "enabled": False,
                    },
                },
            }
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(config, f, sort_keys=False)

            real_create_study = run_optuna_experiment.optuna.create_study

            def create_study_with_fixed_trial_count(*args, **kwargs):
                study = real_create_study(*args, **kwargs)
                real_optimize = study.optimize

                def optimize(objective, *opt_args, **opt_kwargs):
                    opt_kwargs.setdefault("n_trials", 2)
                    return real_optimize(objective, *opt_args, **opt_kwargs)

                study.optimize = optimize
                return study

            with mock.patch.object(run_optuna_experiment, "TABLE_PATH", str(table_path)):
                with mock.patch.object(
                    run_optuna_experiment.optuna,
                    "create_study",
                    side_effect=create_study_with_fixed_trial_count,
                ):
                    with mock.patch.object(
                        sys,
                        "argv",
                        ["run_optuna_experiment.py", str(config_path)],
                    ):
                        old_cwd = os.getcwd()
                        try:
                            os.chdir(repo_root)
                            run_optuna_experiment.main()
                        finally:
                            os.chdir(old_cwd)

            experiment_roots = list(output_root.iterdir())
            matching_roots = sorted(
                path for path in experiment_roots
                if path.name.startswith("tiny_optuna_gpu_")
            )
            self.assertTrue(matching_roots)
            experiment_root = matching_roots[-1]

            study_summary = json.loads((experiment_root / "study_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(2, study_summary["num_trials"])
            self.assertEqual(2, study_summary["num_completed_trials"])
            self.assertEqual("study_timeout", study_summary["stop_reason"])
            self.assertIsNotNone(study_summary["best_params"])
            self.assertEqual("train_loss", study_summary["selection_metric"])
            self.assertIn("best_train_value", study_summary)
            self.assertIn("best_test_value", study_summary)
            self.assertNotIn("train_target", study_summary)
            self.assertNotIn("test_target", study_summary)
            self.assertTrue(Path(study_summary["all_records_path"]).exists())

            trials = [
                json.loads(line)
                for line in (experiment_root / "trials.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(2, len(trials))

            for trial in trials:
                self.assertEqual(0, trial["returncode"])
                self.assertTrue(Path(trial["summary_path"]).exists())
                self.assertTrue(Path(trial["log_path"]).exists())
                self.assertTrue(Path(trial["records_path"]).exists())
                self.assertEqual("train_loss", trial["selection_metric"])

                summary = json.loads(Path(trial["summary_path"]).read_text(encoding="utf-8"))
                self.assertEqual("max_iters_reached", summary["termination_reason"])
                self.assertIn("best_train_loss", summary)
                self.assertIn("best_val_loss", summary)

                log_text = Path(trial["log_path"]).read_text(encoding="utf-8")
                self.assertIn("step 0: train loss", log_text)
                self.assertIn("step 1: train loss", log_text)

                records = [
                    json.loads(line)
                    for line in Path(trial["records_path"]).read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                self.assertTrue(records)
                self.assertEqual("stage1", records[0]["stage"])
                self.assertEqual(trial["trial_id"], records[0]["trial_id"])

            all_records = [
                json.loads(line)
                for line in (experiment_root / "all_records.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertTrue(all_records)
            self.assertEqual({"stage1"}, {record["stage"] for record in all_records})


class LearningRateLoadTest(unittest.TestCase):
    def test_load_learning_rate_from_checkpoint_uses_optimizer_param_group(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmpdir_str:
            checkpoint_path = Path(tmpdir_str) / "ckpt.pt"
            torch.save(
                {
                    "optimizer": {
                        "param_groups": [
                            {"lr": 1.234e-4},
                        ]
                    }
                },
                checkpoint_path,
            )

            loaded = run_optuna_experiment.load_learning_rate_from_checkpoint(str(checkpoint_path))

            self.assertEqual(1.234e-4, loaded)

    def test_load_learning_rate_from_run_prefers_checkpoint_over_summary_value(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmpdir_str:
            run_dir = Path(tmpdir_str)
            checkpoint_path = run_dir / "ckpt_last.pt"
            torch.save(
                {
                    "optimizer": {
                        "param_groups": [
                            {"lr": 7.89e-5},
                        ]
                    }
                },
                checkpoint_path,
            )
            summary = {
                "best_checkpoint_path": "",
                "last_checkpoint_path": str(checkpoint_path),
                "learning_rate": 1e-6,
            }

            loaded = run_optuna_experiment.load_learning_rate_from_run(
                summary=summary,
                run_dir=str(run_dir),
            )

            self.assertEqual(7.89e-5, loaded)


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import yaml

import run_stage1_optuna


def make_config(output_root):
    return {
        "experiment": {
            "name": "llama60m_lr_search",
            "train_script": "train.py",
            "output_root": str(output_root),
            "target_family": "LLAMA",
            "target_dataset": "C4",
            "target_model_size": "60m",
        },
        "hyperparameters": {
            "learning_rate": {
                "type": "log_uniform",
                "range": [1e-5, 1e-3],
            }
        },
        "task": {
            "train_metric": "train_loss",
            "test_metric": "val_loss",
            "metric_mode": "min",
            "num_iterations_per_trial": 100,
            "max_running_time_per_trial_hours": 1.0,
        },
        "fixed_args": {
            "max_iters": 100,
            "lr_decay_iters": 100,
        },
        "checkpoint": {
            "save_last": True,
        },
    }


def make_candidate_plan(count):
    ordered_candidates = []
    for index in range(count):
        ordered_candidates.append(
            {
                "candidate_rank": index,
                "candidate_index": index,
                "params": {
                    "learning_rate": 1e-4 + index * 1e-5,
                },
            }
        )
    return {
        "tuned_param_name": "learning_rate",
        "lr_values": [candidate["params"]["learning_rate"] for candidate in ordered_candidates],
        "scheduler_values": ["cosine_10pct"],
        "full_candidate_count": count,
        "sample_size": count,
        "sample_seed": 1337,
        "ordered_candidates": ordered_candidates,
    }


def make_args(config_path, batch_index=None):
    return types.SimpleNamespace(
        config=str(config_path),
        batch_index=batch_index,
        num_trials=12,
    )


class RunStage1OptunaBatchProtocolTest(unittest.TestCase):
    def test_initialize_controller_state_uses_first_12_candidates_in_3_batches(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            config_path = tmpdir / "config.yaml"
            config = make_config(tmpdir)
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            args = make_args(config_path)

            with mock.patch.object(
                run_stage1_optuna,
                "get_or_create_persisted_candidate_plan",
                return_value=make_candidate_plan(16),
            ):
                state = run_stage1_optuna.initialize_controller_state(
                    run_root=str(tmpdir / "run"),
                    args=args,
                    config=config,
                    tuning_iters=25,
                    total_training_iters=100,
                )

            self.assertEqual(12, len(state["trials"]))
            self.assertEqual(3, state["batch_count"])
            self.assertEqual(4, state["batch_size"])
            self.assertEqual(25, state["tuning_iters"])
            self.assertEqual(
                [f"trial_{index:04d}" for index in range(12)],
                [trial["trial_id"] for trial in state["trials"]],
            )
            self.assertEqual(
                [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
                [trial["batch_index"] for trial in state["trials"]],
            )

    def test_initialize_controller_state_can_select_only_middle_batch(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            config_path = tmpdir / "config.yaml"
            config = make_config(tmpdir)
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            args = make_args(config_path, batch_index=2)

            with mock.patch.object(
                run_stage1_optuna,
                "get_or_create_persisted_candidate_plan",
                return_value=make_candidate_plan(16),
            ):
                state = run_stage1_optuna.initialize_controller_state(
                    run_root=str(tmpdir / "run"),
                    args=args,
                    config=config,
                    tuning_iters=25,
                    total_training_iters=100,
                )

            self.assertEqual([2], state["requested_batch_indices"])
            self.assertEqual([1], state["batch_indices_to_run"])
            self.assertEqual(4, len(state["trials"]))
            self.assertEqual(
                [f"trial_{index:04d}" for index in range(4, 8)],
                [trial["trial_id"] for trial in state["trials"]],
            )
            self.assertEqual(
                [1, 1, 1, 1],
                [trial["batch_index"] for trial in state["trials"]],
            )

    def test_initialize_controller_state_rejects_fewer_than_12_candidates(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            config_path = tmpdir / "config.yaml"
            config = make_config(tmpdir)
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            args = make_args(config_path)

            with mock.patch.object(
                run_stage1_optuna,
                "get_or_create_persisted_candidate_plan",
                return_value=make_candidate_plan(11),
            ):
                with self.assertRaisesRegex(ValueError, "requires at least 12 ordered candidates"):
                    run_stage1_optuna.initialize_controller_state(
                        run_root=str(tmpdir / "run"),
                        args=args,
                        config=config,
                        tuning_iters=25,
                        total_training_iters=100,
                    )

    def test_run_batch_runs_four_tuning_trials_and_one_resumed_winner(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            config_path = tmpdir / "config.yaml"
            config = make_config(tmpdir)
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            args = make_args(config_path)
            phase_calls = []

            tuning_values = {
                "trial_0000": 4.0,
                "trial_0001": 3.0,
                "trial_0002": 1.0,
                "trial_0003": 2.0,
            }
            final_values = {
                "trial_0002": 0.7,
            }

            def fake_run_trial_phase(config, phase_name, batch_index, target_iters, trial_state, batch_root, all_records_path):
                phase_calls.append((phase_name, trial_state["trial_id"], target_iters))
                trial_dir = Path(trial_state["trial_dir"])
                summary_path = trial_dir / "summary.json"
                records_path = trial_dir / "records.jsonl"
                log_path = trial_dir / "trial.log"
                if phase_name == "tuning":
                    train_value = tuning_values[trial_state["trial_id"]]
                    val_value = train_value + 0.1
                    (trial_dir / "ckpt_last.pt").write_text("checkpoint", encoding="utf-8")
                else:
                    train_value = final_values[trial_state["trial_id"]]
                    val_value = final_values[trial_state["trial_id"]]
                summary = {
                    "best_train_loss": train_value,
                    "best_val_loss": val_value,
                    "iter_num": target_iters,
                    "termination_reason": "max_iters_reached",
                    "elapsed_wall_clock_hours": 0.5,
                    "forward_backward_hours": 0.4,
                    "train_script": "train.py",
                    "last_checkpoint_path": str(trial_dir / "ckpt_last.pt"),
                }
                summary_path.write_text(json.dumps(summary), encoding="utf-8")
                records_path.write_text("{}\n", encoding="utf-8")
                log_path.write_text("step 1: train loss 1.0, val loss 1.1\n", encoding="utf-8")

                record = {
                    "trial_id": trial_state["trial_id"],
                    "trial_number": trial_state["trial_number"],
                    "batch_index": batch_index,
                    "params": trial_state["params"],
                    "selection_metric": config["task"]["train_metric"],
                    "train_objective_value": train_value,
                    "test_objective_value": val_value,
                    "returncode": 0,
                    "summary_path": str(summary_path),
                    "trial_dir": str(trial_dir),
                    "log_path": str(log_path),
                    "records_path": str(records_path),
                    "termination_reason": "max_iters_reached",
                    "phase": phase_name,
                    "target_iters": target_iters,
                    "completed_iters": target_iters,
                    "init_from": "resume" if phase_name == "full_training" else "scratch",
                }
                trial_state["last_summary"] = summary
                trial_state["completed_iters"] = target_iters
                if phase_name == "tuning":
                    trial_state["tuning_completed"] = True
                    trial_state["tuning_objective_value"] = train_value
                    trial_state["tuning_test_value"] = val_value
                    trial_state["tuning_record"] = record
                else:
                    trial_state["full_completed"] = True
                    trial_state["final_objective_value"] = train_value
                    trial_state["final_test_value"] = val_value
                    trial_state["full_record"] = record
                return record

            with mock.patch.object(
                run_stage1_optuna,
                "get_or_create_persisted_candidate_plan",
                return_value=make_candidate_plan(12),
            ):
                state = run_stage1_optuna.initialize_controller_state(
                    run_root=str(tmpdir / "run"),
                    args=args,
                    config=config,
                    tuning_iters=25,
                    total_training_iters=100,
                )

            with mock.patch.object(run_stage1_optuna, "run_trial_phase", side_effect=fake_run_trial_phase):
                result = run_stage1_optuna.run_batch(
                    config=config,
                    config_path=str(config_path),
                    state=state,
                    batch_index=0,
                    session_start_time=0.0,
                )

            self.assertEqual(
                [
                    ("tuning", "trial_0000", 25),
                    ("tuning", "trial_0001", 25),
                    ("tuning", "trial_0002", 25),
                    ("tuning", "trial_0003", 25),
                    ("full_training", "trial_0002", 100),
                ],
                phase_calls,
            )
            self.assertEqual("trial_0002", result["selected_trial_id"])
            self.assertTrue(state["trials"][2]["full_completed"])
            self.assertFalse(state["trials"][0]["full_completed"])
            self.assertFalse(state["trials"][1]["full_completed"])
            self.assertFalse(state["trials"][3]["full_completed"])

    def test_run_batch_uses_completed_winner_final_losses_for_cumulative_best(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            config_path = tmpdir / "config.yaml"
            config = make_config(tmpdir)
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            args = make_args(config_path)

            tuning_values = {
                "trial_0000": 10.0,
                "trial_0001": 11.0,
                "trial_0002": 12.0,
                "trial_0003": 13.0,
                "trial_0004": 1.0,
                "trial_0005": 2.0,
                "trial_0006": 3.0,
                "trial_0007": 4.0,
            }
            final_values = {
                "trial_0000": 0.8,
                "trial_0004": 0.9,
            }

            def fake_run_trial_phase(config, phase_name, batch_index, target_iters, trial_state, batch_root, all_records_path):
                trial_dir = Path(trial_state["trial_dir"])
                summary_path = trial_dir / "summary.json"
                records_path = trial_dir / "records.jsonl"
                log_path = trial_dir / "trial.log"
                if phase_name == "tuning":
                    train_value = tuning_values[trial_state["trial_id"]]
                    val_value = train_value + 0.01
                    (trial_dir / "ckpt_last.pt").write_text("checkpoint", encoding="utf-8")
                else:
                    train_value = final_values[trial_state["trial_id"]]
                    val_value = final_values[trial_state["trial_id"]]
                summary = {
                    "best_train_loss": train_value,
                    "best_val_loss": val_value,
                    "iter_num": target_iters,
                    "termination_reason": "max_iters_reached",
                    "elapsed_wall_clock_hours": 0.5,
                    "forward_backward_hours": 0.4,
                    "train_script": "train.py",
                    "last_checkpoint_path": str(trial_dir / "ckpt_last.pt"),
                }
                summary_path.write_text(json.dumps(summary), encoding="utf-8")
                records_path.write_text("{}\n", encoding="utf-8")
                log_path.write_text("step 1: train loss 1.0, val loss 1.1\n", encoding="utf-8")
                record = {
                    "trial_id": trial_state["trial_id"],
                    "trial_number": trial_state["trial_number"],
                    "batch_index": batch_index,
                    "params": trial_state["params"],
                    "selection_metric": config["task"]["train_metric"],
                    "train_objective_value": train_value,
                    "test_objective_value": val_value,
                    "returncode": 0,
                    "summary_path": str(summary_path),
                    "trial_dir": str(trial_dir),
                    "log_path": str(log_path),
                    "records_path": str(records_path),
                    "termination_reason": "max_iters_reached",
                    "phase": phase_name,
                    "target_iters": target_iters,
                    "completed_iters": target_iters,
                    "init_from": "resume" if phase_name == "full_training" else "scratch",
                }
                trial_state["last_summary"] = summary
                trial_state["completed_iters"] = target_iters
                if phase_name == "tuning":
                    trial_state["tuning_completed"] = True
                    trial_state["tuning_objective_value"] = train_value
                    trial_state["tuning_test_value"] = val_value
                    trial_state["tuning_record"] = record
                else:
                    trial_state["full_completed"] = True
                    trial_state["final_objective_value"] = train_value
                    trial_state["final_test_value"] = val_value
                    trial_state["full_record"] = record
                return record

            with mock.patch.object(
                run_stage1_optuna,
                "get_or_create_persisted_candidate_plan",
                return_value=make_candidate_plan(12),
            ):
                state = run_stage1_optuna.initialize_controller_state(
                    run_root=str(tmpdir / "run"),
                    args=args,
                    config=config,
                    tuning_iters=25,
                    total_training_iters=100,
                )

            with mock.patch.object(run_stage1_optuna, "run_trial_phase", side_effect=fake_run_trial_phase):
                batch1 = run_stage1_optuna.run_batch(
                    config=config,
                    config_path=str(config_path),
                    state=state,
                    batch_index=0,
                    session_start_time=0.0,
                )
                batch2 = run_stage1_optuna.run_batch(
                    config=config,
                    config_path=str(config_path),
                    state=state,
                    batch_index=1,
                    session_start_time=0.0,
                )

            self.assertEqual("trial_0000", batch1["selected_trial_id"])
            self.assertEqual("trial_0004", batch2["selected_trial_id"])
            self.assertEqual(0.8, batch2["cumulative_best_completed_winner_so_far"]["winner_final_loss"])
            self.assertEqual("trial_0000", batch2["cumulative_best_completed_winner_so_far"]["winner_trial_id"])

    def test_run_batch_rejects_missing_resume_checkpoint_for_winner(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            config_path = tmpdir / "config.yaml"
            config = make_config(tmpdir)
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            args = make_args(config_path)

            def fake_run_trial_phase(config, phase_name, batch_index, target_iters, trial_state, batch_root, all_records_path):
                trial_dir = Path(trial_state["trial_dir"])
                summary_path = trial_dir / "summary.json"
                records_path = trial_dir / "records.jsonl"
                log_path = trial_dir / "trial.log"
                train_value = float(trial_state["trial_number"])
                summary = {
                    "best_train_loss": train_value,
                    "best_val_loss": train_value,
                    "iter_num": target_iters,
                    "termination_reason": "max_iters_reached",
                    "elapsed_wall_clock_hours": 0.5,
                    "forward_backward_hours": 0.4,
                    "train_script": "train.py",
                    "last_checkpoint_path": "",
                }
                summary_path.write_text(json.dumps(summary), encoding="utf-8")
                records_path.write_text("{}\n", encoding="utf-8")
                log_path.write_text("step 1: train loss 1.0, val loss 1.1\n", encoding="utf-8")
                record = {
                    "trial_id": trial_state["trial_id"],
                    "trial_number": trial_state["trial_number"],
                    "batch_index": batch_index,
                    "params": trial_state["params"],
                    "selection_metric": config["task"]["train_metric"],
                    "train_objective_value": train_value,
                    "test_objective_value": train_value,
                    "returncode": 0,
                    "summary_path": str(summary_path),
                    "trial_dir": str(trial_dir),
                    "log_path": str(log_path),
                    "records_path": str(records_path),
                    "termination_reason": "max_iters_reached",
                    "phase": phase_name,
                    "target_iters": target_iters,
                    "completed_iters": target_iters,
                    "init_from": "scratch",
                }
                trial_state["last_summary"] = summary
                trial_state["completed_iters"] = target_iters
                trial_state["tuning_completed"] = True
                trial_state["tuning_objective_value"] = train_value
                trial_state["tuning_test_value"] = train_value
                trial_state["tuning_record"] = record
                return record

            with mock.patch.object(
                run_stage1_optuna,
                "get_or_create_persisted_candidate_plan",
                return_value=make_candidate_plan(12),
            ):
                state = run_stage1_optuna.initialize_controller_state(
                    run_root=str(tmpdir / "run"),
                    args=args,
                    config=config,
                    tuning_iters=25,
                    total_training_iters=100,
                )

            with mock.patch.object(run_stage1_optuna, "run_trial_phase", side_effect=fake_run_trial_phase):
                with self.assertRaisesRegex(FileNotFoundError, "has no resumable checkpoint"):
                    run_stage1_optuna.run_batch(
                        config=config,
                        config_path=str(config_path),
                        state=state,
                        batch_index=0,
                        session_start_time=0.0,
                    )


if __name__ == "__main__":
    unittest.main()

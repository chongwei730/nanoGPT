# Experiment Protocol

This document summarizes the current experiment process for this repository.
Read this together with `docs/research.md` before running any experiment.

## Goal

The research goal is to test whether a line-search-based learning rate scheduler can
adaptively tune learning rate across different model sizes, optimizers, and model
architectures. The main comparison is the wall-clock time required to go from
hyperparameter search to reaching a preset target metric.

## Core Rule

Only tune the learning rate. Do not change any other hyperparameters unless there is
an explicit instruction to do so.

## Hyperparameter Optimization

- Framework: Optuna
- Sampler: TPE (default Optuna sampler)
- Search space: learning rate only

Example config:

```yaml
hyperparameters:
  learning_rate:
    type: log_uniform
    range: [1e-5, 1e-3]
```

All other hyperparameters must stay fixed for a given experiment.

## Task Configuration

Task settings should be defined in the experiment config file under `config/experiments/`.

Expected fields:

```yaml
task:
  train_metric: ...
  test_metric: ...
  target:
    type: threshold / improvement / custom
    value: ...
  max_running_time_per_trial: ...
```

In the current implementation, the target is represented as explicit train and test
thresholds plus a metric mode:

```yaml
task:
  train_metric: train_loss
  test_metric: val_loss
  metric_mode: min
  train_target: ...
  test_target: ...
  max_running_time_per_trial_hours: ...
```

## Trial Execution

Stage one:
Use Optuna to search only the learning rate within the predefined learning rate range.
Each Optuna trial is limited by `task.max_running_time_per_trial_hours`.
All other hyperparameters must remain fixed.
In stage one, always use 5% of the total training tokens as training length of one trial.

Stage two:
Use the best learning rate found by Optuna and run the final training trial.
The reported total experiment time only includes this stage-two final training run,
not the Optuna search time from stage one.

Run Stage1-2 multiple times for different level of `task.max_running_time_per_trial_hours`. 

## Logging

Track the following at each logging or evaluation interval:

- `step`
- `train_loss`
- `train_metric`
- `test_loss`
- `test_metric`
- `learning_rate`
- `wall_clock_time`

## Checkpointing

During training:

- Save the best model checkpoint using the best `test_metric`
- Save the last checkpoint

Protocol form:

```yaml
checkpoint:
  save_best: true
  metric: test_metric
  mode: max / min
```

Current implementation detail:

- Best checkpoint is typically saved as `ckpt.pt`
- Last checkpoint may also be saved when enabled
- Best-model selection follows the configured metric mode

## Time Definition

Time is defined as forward wall-clock time plus backward wall-clock time.

In practice, report elapsed wall-clock time for the full training run and the time at
which the configured target metric is first reached.

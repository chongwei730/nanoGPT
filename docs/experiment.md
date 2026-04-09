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

Default trial loop:

```text
num_trials = 0
max_trials = 5

while target not reached and num_trials < max_trials:
    sample learning_rate with Optuna
    run one training trial
    stop the trial when the target is reached or time limit is hit
    num_trials += 1
```

Operationally:

- Optuna samples one learning rate per trial.
- The selected training script runs with all non-LR hyperparameters fixed.
- Continue using Optuna until the metric target is reached or the trial budget is exhausted.

## Logging

Track the following at each logging or evaluation interval:

- `step`
- `train_loss`
- `train_metric`
- `test_loss`
- `test_metric`
- `learning_rate`
- `wall_clock_time`

For the current training scripts, the most important logged values are:

- step
- train loss
- val loss
- learning rate
- wall-clock time

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

## Trial Termination

A trial ends when one of the following happens:

- `max_running_time_per_trial` is reached
- both of the target conditions are reached (both training and validation target metric)
- the trial is pruned by the Optuna pruner

Timeout is treated as a valid termination.

After each trial:

```text
num_trials += 1
```

## Time Definition

Time is defined as forward wall-clock time plus backward wall-clock time.

In practice, report elapsed wall-clock time for the full training run and the time at
which the configured target metric is first reached.

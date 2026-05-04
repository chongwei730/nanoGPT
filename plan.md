# Tuning Batch Protocol Migration Plan

## Summary

The experiment controller should be updated from a serial successive halving workflow to a fixed tuning-batch workflow.

This protocol is fixed as follows:

- Use exactly 12 hyperparameter combinations.
- Split the 12 combinations into 3 tuning batches.
- Each tuning batch contains exactly 4 combinations.
- In each batch, run all 4 trials to `ceil(total_training_iters / 4)`.
- Select the best trial in that batch using the configured tuning metric.
- Resume only that batch winner from the `1/4` checkpoint to full training.
- After batch 1, batch 2, and batch 3, report the best final loss among completed batch winners so far.

## Goal

Replace the old pruning-based controller with a fixed protocol that evaluates candidates in three ordered 4-trial batches and promotes one winner per batch to full training.

The plan in this document is intended to be decision complete so that an implementation agent can update the controller and related outputs without making any protocol decisions.

## Protocol Definition

### Candidate ordering

- The controller must continue to use the fixed candidate pool.
- The controller must consume the first 12 candidate combinations from that pool.
- The 12 combinations must be assigned in order:
  - Batch 1 uses candidates 1 to 4.
  - Batch 2 uses candidates 5 to 8.
  - Batch 3 uses candidates 9 to 12.

### Tuning budget

- Let `total_training_iters` be the full training budget for one full run.
- Define `tuning_iters = ceil(total_training_iters / 4)`.
- Every trial in every tuning batch must run from scratch to exactly `tuning_iters`.

### Batch winner selection

- At the end of each 4-trial tuning batch, rank the 4 trials using the configured tuning metric.
- Use the existing experiment metric configuration:
  - `metric_mode = min` means smaller is better.
  - `metric_mode = max` means larger is better.
- If two trials tie on the tuning metric, choose the earlier candidate in the fixed candidate order.

### Continuation to full training

- After each tuning batch, resume only the batch winner.
- The batch winner must continue from its saved checkpoint at `tuning_iters`.
- The resumed winner must run to full training end at `total_training_iters`.
- The three non-winning trials in that batch must not continue.
- Each batch winner must receive its own full run.
- Winners from earlier batches remain completed results and are not resumed again during later batches.

## Reporting Semantics

The phrase "best loss among the 4/8/12 trials" must be interpreted using completed batch winners only.

- After batch 1 completes, report the final loss of the batch 1 winner.
- After batch 2 completes, report the best final loss among the completed winners from batches 1 and 2.
- After batch 3 completes, report the best final loss among the completed winners from batches 1, 2, and 3.

This reporting rule must not use quarter-budget tuning losses for the cumulative 4/8/12 report.

The cumulative reported result after each batch must therefore be based on:

- completed full runs only
- one completed winner per finished batch
- the final best loss from each completed winner

## Required Implementation Changes

### Controller logic

- Replace rung-based survivor pruning with fixed batch execution.
- Remove dependency on halving-style rung counts for this protocol.
- Enforce exactly 3 tuning batches and exactly 4 trials per batch.
- Enforce exactly 12 consumed candidate combinations from the fixed candidate pool.

### Controller state

The controller state should record batch-oriented progress instead of rung-oriented progress.

It must explicitly track:

- protocol identifier for this tuning-batch workflow
- total candidate count used by the workflow
- batch count
- batch size
- `tuning_iters`
- completed batches
- batch winner trial IDs
- completed full runs
- current best completed winner so far

### Trial artifacts and resume behavior

- Each trial should continue to use a persistent shared trial directory.
- Every tuning trial must save a resumable checkpoint at the end of its quarter-budget run.
- The winner of each batch must resume from that checkpoint for the full run.
- Resume logic must not depend on a best checkpoint only; it must work with the required resumable checkpoint.

### Output and result files

- Preserve the existing machine-readable result entry points expected by downstream scripts.
- Replace rung-oriented summaries with batch-oriented summaries.
- Emit one result entry per completed batch.
- Each batch result must include:
  - batch index
  - candidate range used in that batch
  - batch trial IDs
  - tuning iteration budget
  - selected batch winner
  - winner hyperparameters
  - winner tuning metric
  - winner final loss
  - cumulative best completed winner so far
  - cumulative total running time

### Fixed candidate pool usage

- The candidate pool remains the source of ordered hyperparameter combinations.
- The implementation must not resample candidates for this workflow.
- The implementation must fail fast if fewer than 12 ordered candidates are available.

### Downstream compatibility

- Preserve the top-level output contract used by summary and plotting scripts.
- If legacy filenames are still required by downstream readers, continue writing them while changing the payload contents to batch semantics.

## Test And Validation Expectations

Implementation should be validated with controller-level tests that cover the following:

- The controller slices the first 12 candidates into exactly 3 ordered batches of 4.
- Every tuning trial runs only to `tuning_iters`.
- Exactly one winner is selected per batch.
- Each batch winner resumes from checkpoint and runs to `total_training_iters`.
- Non-winning trials do not continue beyond the tuning stage.
- Batch 2 and batch 3 use the next ordered candidates instead of reusing earlier batch candidates.
- The cumulative report after batch 1, 2, and 3 uses completed winner final losses only.
- Total running time is accumulated correctly across tuning and resumed full runs.
- The implementation rejects invalid inputs:
  - fewer than 12 available candidates
  - wrong batch size assumptions
  - missing resumable checkpoint for a selected winner
  - protocol invocation with trial counts that do not match this fixed workflow

## Assumptions

- `plan.md` is created at the repository root.
- This protocol is fixed to 12 candidates, 3 batches, and batch size 4.
- `tuning_iters = ceil(total_training_iters / 4)`.
- Each batch winner receives an independent full run.
- Final comparison is across completed batch winners only.
- This step is documentation only and does not change controller code yet.

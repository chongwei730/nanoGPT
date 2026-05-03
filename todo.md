# TODO

## Modification Plan for Monotone Scheduler + LR Tuning

Implement the following experiment-setup changes.

### 1. Expand the hyperparameter space

- Add scheduler choice as a tuned hyperparameter in addition to learning rate.
- Restrict scheduler options to exactly these three choices:
  - cosine decay to 10% of the initial learning rate
  - inverse square root
  - linear decay to 10% of the initial learning rate
- Change the lower bound of the learning-rate search space from `1e-5` to `1e-6`.
- Keep the upper bound unchanged unless there is a separate explicit instruction.

### 2. Replace free-form sampling with a fixed discrete candidate pool

- Generate exactly 50 discrete learning-rate values from the configured search range.
- Form the full grid of scheduler and learning-rate choices, giving `50 * 3 = 150`
  total hyperparameter combinations.
- Sample exactly 16 combinations from this 150-combination pool once.
- Freeze the sampled 16-combination list and its order.
- Reuse this same ordered list for every tuning run so different trial budgets are
  directly comparable.

### 3. Enforce monotone tuning behavior across budgets

- When a tuning run uses budget `N`, run the first `N` combinations from the fixed
  ordered list sequentially.
- Use budgets that are powers of 2, such as 4, 8, and 16.
- The implementation must preserve the prefix property:
  - the 8-trial run must include the exact first 4 combinations from the 4-trial run
  - the 16-trial run must include the exact first 8 combinations from the 8-trial run
- This is required so larger tuning budgets always achieve a result that is better
  than or equal to smaller budgets, assuming selection is based on the best completed
  combination among those evaluated.
- This also makes total tuning time monotone with respect to the chosen budget.

### 4. Update serial successive halving settings

- Change the halving ratio from 4 to 2.
- Update every place in the controller logic that assumes reduction factor 4.
- Verify that rung construction, survivor counts, and resume behavior all match the
  new factor-2 schedule.

### 5. Remove the scheduler-free Adam baseline

- Remove the scheduler-free Adam baseline from the experiment setup.
- Remove it from any result aggregation, summary-table labeling, or comparison logic
  that still expects it to exist.

### 6. Implementation constraints

- Keep the rest of the experiment protocol unchanged unless required by the changes
  above.
- Do not introduce extra tuned hyperparameters.
- Preserve existing output structure and resume/checkpoint behavior where possible.
- Any test command prepared for this work must stay close to the real experiment
  setting, only shortened in runtime.

## Research Context

- Always read docs/research.md before running experiments
- Follow docs/experiment.md settings strictly
- Do not introduce new variables unless necessary
- Unless explicitly instructed, do not modify Markdown files or 
- Every time when asked for a test, the test need to be close to the real setting. for example, if if the real setting it need to be run on multi GPUs, the test need to be run in multi gpus. Don't run any test by yourself, Instead, output commands to run tests. The test should be all the same as the actual run, except it takes much less time to run. All the essential part: data load, training, eval, all parts in the experiment protocol, and output storage, should be the same as actual run.
contents under docs/


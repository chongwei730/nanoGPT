#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARTITIONS="preempt-gpu,a100-8,a100-4,apollo_agate"

declare -a JOB_SCRIPTS=(
  "cosine/num_trials_8.sh"
  # "cosine/num_trials_12.sh"
  # "cosine/num_trials_16.sh"
  "muon/num_trials_8.sh"
  # "muon/num_trials_12.sh"
  # "muon/num_trials_16.sh"
  "schedulefree_adam/num_trials_8.sh"
  # "schedulefree_adam/num_trials_12.sh"
  # "schedulefree_adam/num_trials_16.sh"
  "line_search.sh"
  "line_search_muon.sh"
)

for rel_path in "${JOB_SCRIPTS[@]}"; do
  job_script="${SCRIPT_DIR}/${rel_path}"
  echo "Submitting 5-minute test for ${job_script}"
  sbatch --partition="${PARTITIONS}" --time=00:05:00 "${job_script}"
done

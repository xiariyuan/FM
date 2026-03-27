#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

patterns=(
  "summary.csv"
  "result.csv"
  "metrics.csv"
  "metrics.jsonl"
  "*.metrics.jsonl"
  "report.md"
  "summary.json"
  "sequence_cluster_summary.csv"
  "family_runs.csv"
)

files=()
for pattern in "${patterns[@]}"; do
  while IFS= read -r -d '' file; do
    files+=("${file}")
  done < <(find outputs -type f -name "${pattern}" -print0)
done

if [[ -f "outputs/experiment_registry.csv" ]]; then
  files+=("outputs/experiment_registry.csv")
fi

if [[ ${#files[@]} -eq 0 ]]; then
  echo "No structured experiment record files found under outputs/."
  exit 0
fi

git add .gitignore
git add -- "${files[@]}"

echo "Staged ${#files[@]} structured experiment record files."

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

REMOTE="${FMTRACK_GIT_REMOTE:-origin}"
BRANCH="${FMTRACK_GIT_BRANCH:-$(git rev-parse --abbrev-ref HEAD)}"
DEFAULT_SSH_KEY="/root/.ssh/fmtrack_repo_push_key"

if [[ -z "${FMTRACK_GIT_SSH_KEY:-}" && -f "${DEFAULT_SSH_KEY}" ]]; then
  FMTRACK_GIT_SSH_KEY="${DEFAULT_SSH_KEY}"
fi

if [[ -n "${FMTRACK_GIT_SSH_KEY:-}" ]]; then
  export GIT_SSH_COMMAND="ssh -i ${FMTRACK_GIT_SSH_KEY} -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new"
fi

"${ROOT_DIR}/scripts/git_stage_experiment_records.sh"

if git diff --cached --quiet; then
  echo "No new structured experiment record changes to commit."
  exit 0
fi

if [[ $# -gt 0 ]]; then
  COMMIT_MSG="$*"
else
  COMMIT_MSG="Sync experiment records $(date '+%Y-%m-%d %H:%M:%S %z')"
fi

git commit -m "${COMMIT_MSG}"
git push "${REMOTE}" "${BRANCH}"

echo "Committed and pushed structured experiment records to ${REMOTE}/${BRANCH}."

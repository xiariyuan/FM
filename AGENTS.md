# Agent Rules

## Experiment Recording

Every experiment must be recorded.

Mandatory requirements:
- Each experiment run must have a local structured record file such as `result.csv`, `summary.csv`, `metrics.csv`, or `*.metrics.jsonl`.
- Queue-style experiment batches must maintain a queue-level `summary.csv`.
- Important experiment batches must also be appended to the central registry at `outputs/experiment_registry.csv`.
- If an experiment is currently running, its structured record must show `status=running` instead of leaving an old `failed` or stale status in place.
- If a run is restarted manually, the corresponding CSV record must be updated or backfilled immediately.
- Do not rely on raw logs alone as the only experiment record.

Operational expectation:
- Before reporting status to the user, check the real running process and the latest structured record together.
- Before ending a long queue, ensure the produced results have been written to CSV or JSONL and are traceable from the registry.

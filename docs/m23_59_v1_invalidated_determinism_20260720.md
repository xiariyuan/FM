# M23-59 v1 invalidated: CUDA determinism implementation error

## Status

M23-59 v1 is invalid and closed. No v1 metric, checkpoint, feature dump, synthetic example bank, threshold, or calibration artifact may be used for scientific gating or final reporting.

## Trigger

The first external-pretraining process emitted a PyTorch warning that CuBLAS `Linear` operations were not deterministic because `CUBLAS_WORKSPACE_CONFIG` had not been set before CUDA initialization. The v1 script requested deterministic algorithms with `warn_only=True`, so the process could continue despite violating the frozen deterministic-training declaration.

## Protocol consequence

MOT17 labels had already been opened. Following the preregistered error policy, the training process was terminated, the log was retained, v1 was declared wholly invalid, and no partial resume or v1 artifact reuse is permitted. No external checkpoint was produced; no MOT20 label was opened; no outer or COMBINED TrackEval was run.

## Evidence

- v1 script SHA-256: `9e67c9fedea18f6e72929569b288a5c691b09353f4d1b1cdba36d6b492255327`
- v1 preregistration JSON SHA-256: `258b06b945dc90fe40d1fbc50152509115bcf182fb302abd3a73431be798b59a`
- v1 implementation manifest SHA-256: `f895b7638c11aca1b3ec9d9cfc5c3d97364b8fa1956943f6205fd4a557a8925c`
- stopped process log SHA-256: `7bddc7c85dbbb8184d7c3332363647881b43ee14eefaf711ebf8ab3a4009d9c6`
- invalidation manifest SHA-256: `32f19ee6e20d266edd648132bd131ddf9df42add69cd719dd0bcd34dbb91ca3f`

## v2-only fix scope

The scientific protocol is unchanged. v2 only sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` before importing `torch`, changes deterministic algorithms from warning-only to hard failure, disables CUDA/CuDNN TF32, changes all versioned output and manifest paths to v2, and reruns every stage from raw data in an empty v2 root.

# MOT20 M23-26 Test Submission Freeze

Date: 2026-07-19

## Submission package

- Path: `outputs/mot20_m23_20260718/m23_26_test_deploy_oof_ensemble_v1/MOT20_M23_26_OOF_ensemble_test_submission.zip`
- Size: 9,603,126 bytes
- SHA-256: `60d1e6dcd24b820850bfff457dc88352ebb4bb207dcdeb248eaa07604a8b32ff`
- MOT20 profile: `mot20_test_4`
- Independent package precheck: PASS
- Root entries: `MOT20-04.txt`, `MOT20-06.txt`, `MOT20-07.txt`, `MOT20-08.txt`

## Deployment protocol

- Protocol: OOF train transaction model plus LOSO-weight-averaged FastReID; MOT20 test inference is GT-free.
- Test GT read: false.
- Parent test tracker: `outputs/assocriskbench_p15_20260713/identity_debt_a43_test_submission/track_results`
- Train parent tracker: `outputs/assocriskbench_p15_20260714/fused_a42_adaptive_risk_eval/l1_r2_q75_adaptive_priority/track_results`
- Chosen policy: loss multiplier `16.0`, minimum positive probability `0.9`, score quantile `0.999`.
- Averaged checkpoint SHA-256: `1f32c4db62a5e9b5950c5a167c5eeab6620dbf9531956d8bc6d4d32ed4b20d8e`
- Four strict-LOSO checkpoints were averaged over compatible inference tensors. The fold-specific classifier head was excluded from inference.

## Sequence outputs

| Sequence | Rows | Selected actions | Output SHA-256 |
|---|---:|---:|---|
| MOT20-04 | 341,208 | 28 | `94164b04b7be900aab3550a1eee41db3f509edea0eebe89d836108e93e458238` |
| MOT20-06 | 147,191 | 15 | `cf5c59cb61d864959c13c2342c839a453ffa24d41801ee66bb2cb0327b3217bf` |
| MOT20-07 | 33,061 | 6 | `d30d2339600514792ed2b95a5daac0cc0f5db7f94f54e5cc82167c8257b16416` |
| MOT20-08 | 99,118 | 11 | `16cc5a78fd3081e240d2f5d1150bf92624f8955a298bc8c44b0c96d3866839be` |

## Validation

`python scripts/check_mot20_submission.py --zip-path <zip> --profile mot20_test_4`

Result:

```text
Expected txt files: 4
Found root txt files: 4
[PASS] Submission package meets the selected profile.
```

The submission ZIP itself remains in the experiment output directory and is intentionally not committed to Git.

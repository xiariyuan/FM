# M23-25 strict sequence-LOSO final result (2026-07-18)

## Final decision

- Protocol: four strict outer-held trackers concatenated before one TrackEval.
- Outer-held GT use: final TrackEval only.
- Final COMBINED HOTA: **79.025010**.
- DetA: **81.565160**.
- AssA: **76.614870**.
- IDSW: **974**.
- Exploratory MOT20 test submission gate (`HOTA > 79.000000`): **passed by +0.025010**.
- Research target (`HOTA > 80.000000`): **not reached; gap 0.974990**.
- Queue status: completed and exited normally.

The `promoted=false` field in the queue report refers to the 80.0 research target, not the separately preregistered 79.0 exploratory test-submission gate.

## Outer-fold results

| Held sequence | HOTA | DetA | AssA | IDSW | Delta HOTA | Selected actions |
|---|---:|---:|---:|---:|---:|---:|
| MOT20-01 | 79.181480 | 81.897146 | 76.711800 | 43 | +0.362780 | 4 |
| MOT20-02 | 72.868943 | 80.713767 | 65.885675 | 317 | +1.294033 | 18 |
| MOT20-03 | 80.571616 | 81.183845 | 79.997680 | 139 | -0.093549 | 0 |
| MOT20-05 | 79.653203 | 81.955170 | 77.457994 | 475 | +0.161763 | 28 |

## Frozen evidence

- Queue summary: `outputs/mot20_m23_20260718/m23_25_loso_queue_v1/summary.csv`
- Combined metrics: `outputs/mot20_m23_20260718/m23_25_sequence_calibrated_graph_loso/combined_oof/metrics.csv`
- Combined report: `outputs/mot20_m23_20260718/m23_25_sequence_calibrated_graph_loso/combined_oof/report.json`
- Combined evaluation log: `outputs/mot20_m23_20260718/m23_25_sequence_calibrated_graph_loso/combined_oof/eval.log`

### Script SHA-256

- `m23_24_train_fastreid_loso.py`: `9423998d2fdc3cf979a1d06a661b7642d9d3b320ff88c6a50cbbe0ee64c57593`
- `m23_24_reembed_phase0.py`: `118d45e7d2bc380c1aed99133130f7b80196a65b6fcd1c1d16343453ae02faa2`
- `m23_24_build_finetuned_micrograph.py`: `5b72b78c2f312e28ce2090793d688108a25742e96084a0395cfdcd57936f3753`
- `m23_25_sequence_calibrated_transaction_graph.py`: `bdd76196627d9ef9f56d35d2b1e392ef3b8202416c70d319fe800aa31bd8d397`
- `m23_25_run_loso_queue.py`: `9800bcd9c19f1c38158df94859049c7a1d64c7794cad35c0d403c7f93f974f0d`

### FastReID checkpoint SHA-256

- MOT20-01: `c828b39144e7a3104faa1d5d6e7c1064ebadd1a9c745c4aae181fb96d9ab2369`
- MOT20-02: `0b2081910e4e4b12c226adb164894546e5119c48b9484ed4cdad2712e86786b4`
- MOT20-03: `c2837ebc435b471d3510bb6457aa3401e1932fbbebdff90d0f9fee6095a1435c`
- MOT20-05: `f7d73fc14486f5913d7e80f32d4745d23163d29bde9329c9735ccfdac90f39d5`

### Outer-held tracker SHA-256

- MOT20-01: `3c8cb3adf76f7de95bde3b96e9ec1c92472ee713e57a1a640e2e99a7552af1a1`
- MOT20-02: `f8d094ad2804ea6a0f4c7d2f5c007529d73f2226df04c69c010300be583d1d7a`
- MOT20-03: `2703ce3722542cb3fbf2492f24a6bf2b49f968491c0eb5f35fa1309028719e9a`
- MOT20-05: `1b80f804e9074a685312bc53ef136237f437903c57e702f37a0f56b8cd0cd887`

### Combined evidence SHA-256

- `metrics.csv`: `cdbff8ce02f3fbaff4650455d932f3239d180d2e03b5880a4025a75354961a70`
- `report.json`: `f7ced1f294c0a832a422467fe5b9dfb4e5d61282f1d957dfe96bbb08d9d06cd8`

## Test-submission safety gate

The train-sequence outer-held tracker files are validation evidence and must not be uploaded as MOT20 test predictions. The result authorizes one exploratory test submission only after a separate GT-free MOT20-test inference run is implemented with the frozen M23-25 method and its test tracker files are audited and packaged. No M23-25-specific test inference/packaging entry point was found under `scripts/m23_research` at freeze time, so no external submission was performed in this checkpoint.

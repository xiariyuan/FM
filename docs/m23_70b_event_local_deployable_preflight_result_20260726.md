# M23-70B Event-Local Deployable Preflight — Result

Decision: **FAIL_EVENT_LOCAL_DEPLOYABLE_REPRESENTATION_AND_ACTION_PRECISION**

No deployable tracker or TrackEval result was produced.

- GT-free event/candidate generation: yes; held labels opened only after candidate parquet freeze.
- Candidate pairs: `3,561,175`; captured teacher actions: `691/972`.
- Event-gate macro AP: `0.171934`; macro recall in top 10% events: `0.739837`.
- Scalar relation ranker macro top-2: `0.686486`.
- Raw 8-frame sequence ranker macro top-2: `0.571548` (worse).
- Full top-20% candidate joint AP after K=3 top-2 restriction: `0.083817`.
- Only 6 actions lie in the 50% precision region; this cannot close the `+0.876807` HOTA gap.
- MOT20 test reads/submissions: `0`.

Next authorized experiment: **M23-70C strict sequence-LOSO metric-adapter representation audit**.

# M23-70C Metric Adapter Representation Gate — Result

Decision: **FAIL_METRIC_ADAPTER_REPRESENTATION_GATE**

- Strict sequence-LOSO folds: `4`.
- Adapter: residual 128D metric adapter, `24,993` parameters.
- Raw candidate top-2 macro recall: `0.423724`.
- Adapted candidate top-2 macro recall: `0.426693`.
- Delta: `+0.002969`.
- MOT20-02 regressed, so the all-fold non-degradation condition failed.
- No tracker was generated and no TrackEval or MOT20 test access occurred.

Conclusion: the frozen 128D appearance representation is the limiting factor; a small post-hoc metric adapter does not recover the missing deployable capacity.

# M23-70B0 Uniform Causal Lattice — Early Close

Decision: **FAIL_GT_FREE_UNIFORM_LATTICE_CAPACITY_AND_EFFICIENCY**

- MOT20-01 HOTA: `78.859127` (`+0.054002` vs M23-46).
- The fixed four-observation lattice recovered almost none of the exact causal-split capacity on MOT20-01.
- MOT20-02 was terminated before candidate-graph freeze after more than seven minutes in GT-free candidate generation; no MOT20-02 teacher stage opened.
- No model training, MOT20 test read, or test submission occurred.
- This mechanical lattice is closed. The next experiment must use event-local branch rollout rather than globally materializing uniform split nodes.

# FGAS-v2 Track-Centric Mainline

## Why v2 exists

`FGAS v1` proved that block-level conflict resolution can help online MOT, but it
still behaved like a local cost-refinement plugin:

- primary-stage only
- scalar edge features only
- MLP over local conflict blocks
- blended back into the original BoT-SORT appearance cost

This is not the same structural level as stronger 2025 association-oriented MOT
methods, which increasingly treat association as a first-class subsystem.

## Mainline decision

Current mainline:

- keep `BoT-SORT` as the carrier for now
- keep `FGAS` as the association-upgrade project
- drop the current frequency branch from mainline priority
- move from `v1 local refiner` to `v2 track-centric block resolver`

Current evidence already supports this:

- `nofreq` online FGAS has positive signal
- current `full` frequency branch is unstable and can turn that signal negative

So `v2` is not "more frequency." It is "more association ownership."

## Implemented tonight

### 1. Stronger resolver architecture

`projects/fgas/fgas/model/block_resolver_v2.py`

`FGASAssociationResolverV2` upgrades the old MLP block scorer into:

- edge encoder
- row token initialization
- column token initialization
- bipartite row-to-column / column-to-row cross-attention
- edge logits
- row no-match logits
- column newborn logits

This keeps the same online IO shape, but the model is now track/detection
token-based instead of edge-only.

### 2. Expanded supervision

`projects/fgas/fgas/data/block_types.py`

Block collation now exports:

- `col_mask`
- `col_newborn_targets`

So training can optimize not only:

- edge match prediction
- row no-match

but also:

- column-side unmatched/newborn behavior

### 3. Training support

`scripts/train_fgas_block_resolver.py`

The trainer now supports:

- `--arch v1`
- `--arch v2_trackdet`

and logs:

- `arch`
- `num_heads`
- `num_attn_layers`
- `col_bce_weight`
- `val_col_bce`

### 4. Runtime support

`projects/fgas/fgas/runtime/block_refiner.py`

Runtime now supports:

- loading `v1` and `v2_trackdet` checkpoints
- `assignment_mode = blend | replace`
- `row_nomatch_weight`

This means the resolver can move closer to owning the local assignment decision
instead of only blending into the original appearance score.

### 5. Queueing

`scripts/queue_fgas_v2_mainline.py`

This queue waits for the current full7 FGAS run to finish, then executes:

1. `smoke_replace`
2. `smoke_blend`
3. pick the better smoke mode
4. `full7_best`

This avoids overnight GPU idle time.

## What is not the mainline anymore

Not mainline:

- current `full` frequency branch as the primary hypothesis
- threshold-only rescue
- going back to the old `FCAA` plugin story

Frequency is now a side signal only. It must re-earn its place later under a
stronger association owner.

## Immediate operating rule

Use `nofreq FGAS-v2` as the current serious line unless a later `full` v2 run
proves otherwise.

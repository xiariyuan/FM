#!/usr/bin/env python3
"""
Test script to verify VAL_SEQUENCES split and logging without training.
Checks:
1. Training dataset excludes VAL_SEQUENCES
2. Validation dataset only includes VAL_SEQUENCES
3. Logs correctly print actual loaded sequences
"""

import sys
from utils.misc import yaml_to_dict
from data import build_dataset

def test_val_split(config_path):
    # Load config
    config = yaml_to_dict(config_path)

    val_sequences = config.get("VAL_SEQUENCES", None)
    if not val_sequences or len(val_sequences) == 0:
        print("ERROR: VAL_SEQUENCES not configured in config!")
        sys.exit(1)

    print(f"Config VAL_SEQUENCES: {val_sequences}")
    print("=" * 80)

    # Build training dataset
    print("\n[1] Building TRAINING dataset (should EXCLUDE VAL_SEQUENCES)...")
    train_dataset = build_dataset(config=config, is_validation=False)
    train_loaded = train_dataset.get_loaded_sequences()

    for dataset_name in train_loaded:
        for split in train_loaded[dataset_name]:
            seqs = train_loaded[dataset_name][split]
            print(f"  [TRAIN] {dataset_name}/{split}: {len(seqs)} sequences")
            print(f"    Sequences: {seqs}")

            # Verify VAL_SEQUENCES are excluded
            # Use substring matching because VAL_SEQUENCES are base names (e.g., MOT17-02)
            # while actual sequences have detector suffixes (e.g., MOT17-02-DPM)
            overlap = [s for s in seqs if any(s.startswith(val_base) for val_base in val_sequences)]
            if overlap:
                print(f"  ❌ ERROR: Training set contains VAL sequences: {overlap}")
                sys.exit(1)
            else:
                print(f"  ✅ PASS: No VAL sequences in training set")

    # Build validation dataset
    print("\n[2] Building VALIDATION dataset (should ONLY include VAL_SEQUENCES)...")
    val_dataset = build_dataset(config=config, is_validation=True)
    val_loaded = val_dataset.get_loaded_sequences()

    for dataset_name in val_loaded:
        for split in val_loaded[dataset_name]:
            seqs = val_loaded[dataset_name][split]
            print(f"  [VAL] {dataset_name}/{split}: {len(seqs)} sequences")
            print(f"    Sequences: {seqs}")

            # For MOT17: each base sequence has 3 detector versions (DPM, FRCNN, SDP)
            # So 'MOT17-02' matches 'MOT17-02-DPM', 'MOT17-02-FRCNN', 'MOT17-02-SDP'
            # Verify all sequences contain at least one VAL base name
            invalid_seqs = [s for s in seqs if not any(s.startswith(val_base) for val_base in val_sequences)]
            if invalid_seqs:
                print(f"  ❌ ERROR: Validation set contains unexpected sequences: {invalid_seqs}")
                sys.exit(1)
            else:
                print(f"  ✅ PASS: All validation sequences match VAL_SEQUENCES base names")


    print("\n" + "=" * 80)
    print("✅ ALL CHECKS PASSED!")
    print("\nSummary:")
    total_train = sum(len(seqs) for dataset in train_loaded.values()
                     for seqs in dataset.values())
    total_val = sum(len(seqs) for dataset in val_loaded.values()
                   for seqs in dataset.values())
    print(f"  Training sequences: {total_train}")
    print(f"  Validation sequences: {total_val}")
    print(f"  Total: {total_train + total_val}")

    return 0

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_val_split.py <config.yaml>")
        sys.exit(1)

    config_path = sys.argv[1]
    sys.exit(test_val_split(config_path))

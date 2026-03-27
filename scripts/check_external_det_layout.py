#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


EXPECTED = {
    "MOT17": [
        "MOT17-02-FRCNN",
        "MOT17-04-FRCNN",
        "MOT17-05-FRCNN",
        "MOT17-09-FRCNN",
        "MOT17-10-FRCNN",
        "MOT17-11-FRCNN",
        "MOT17-13-FRCNN",
    ],
    "MOT20": [
        "MOT20-01",
        "MOT20-02",
        "MOT20-03",
        "MOT20-05",
    ],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", required=True, choices=["sw_yolox", "sgt"])
    ap.add_argument("--dataset", required=True, choices=["MOT17", "MOT20"])
    ap.add_argument("--split", default="train")
    ap.add_argument("--root", default="outputs/external_det")
    args = ap.parse_args()

    base = Path(args.root) / args.detector / args.dataset / args.split
    if not base.exists():
        print(f"[ERR] Missing dir: {base}")
        raise SystemExit(1)

    expected = EXPECTED[args.dataset]
    missing = []
    found = []
    for seq in expected:
        p = base / f"{seq}.txt"
        if p.is_file():
            found.append(seq)
        else:
            missing.append(seq)

    print(f"Detector: {args.detector}")
    print(f"Dataset : {args.dataset}")
    print(f"Split   : {args.split}")
    print(f"Dir     : {base}")
    print(f"Found   : {len(found)}/{len(expected)}")
    if missing:
        print("Missing :")
        for seq in missing:
            print(f"  - {seq}")
    else:
        print("All expected sequence files are present.")

    raise SystemExit(0 if not missing else 2)


if __name__ == "__main__":
    main()

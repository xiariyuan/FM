#!/usr/bin/env python3
"""Monitor RGSA training for failure modes.

Checks defer_rate, reject_rate, rewrite_rate against thresholds from
configs/rgsa_risk_control.yaml and reports warnings.
"""

import argparse
import json
import sys
from pathlib import Path

import yaml


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def check_failure_modes(metrics: dict, config: dict):
    """Check metrics against failure mode thresholds."""
    warnings = []
    failures = []
    thresholds = config.get("failure_modes", {})

    # no_op check
    no_op = thresholds.get("no_op", {}).get("thresholds", {})
    if metrics.get("defer_rate", 0) > no_op.get("defer_rate_max", 0.90):
        failures.append(f"no_op: defer_rate={metrics['defer_rate']:.3f} > {no_op['defer_rate_max']}")

    # idsw_explosion check
    idsw = thresholds.get("idsw_explosion", {}).get("thresholds", {})
    if metrics.get("rewrite_rate", 0) > idsw.get("rewrite_rate_max", 0.30):
        failures.append(f"idsw_explosion: rewrite_rate={metrics['rewrite_rate']:.3f} > {idsw['rewrite_rate_max']}")
    if metrics.get("idsw_increase_pct", 0) > idsw.get("idsw_increase_pct_max", 0.50):
        failures.append(f"idsw_explosion: IDSW increase {metrics['idsw_increase_pct']:.1%}")

    # distribution_shift check
    dist = thresholds.get("distribution_shift", {}).get("thresholds", {})
    for key in ["accept_rate_drift", "defer_rate_drift", "reject_rate_drift"]:
        max_key = f"{key}_max"
        if key in metrics and max_key in dist:
            if abs(metrics[key]) > dist[max_key]:
                warnings.append(f"distribution_shift: {key}={metrics[key]:.3f} > {dist[max_key]}")

    return warnings, failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rgsa_risk_control.yaml")
    parser.add_argument("--metrics-json", required=True, help="Path to metrics JSON from training/eval")
    args = parser.parse_args()

    config = load_config(args.config)
    with open(args.metrics_json) as f:
        metrics = json.load(f)

    warnings, failures = check_failure_modes(metrics, config)

    report = {
        "metrics": metrics,
        "warnings": warnings,
        "failures": failures,
        "safe": len(failures) == 0,
    }

    print(json.dumps(report, indent=2))

    if failures:
        print(f"\n[FAIL] {len(failures)} failure mode(s) detected:")
        for f_msg in failures:
            print(f"  - {f_msg}")
        sys.exit(1)
    elif warnings:
        print(f"\n[WARN] {len(warnings)} warning(s):")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("\n[OK] All metrics within safe bounds.")


if __name__ == "__main__":
    main()

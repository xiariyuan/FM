#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import networkx as nx

MOT20_TEST_SEQS = ["MOT20-04", "MOT20-06", "MOT20-07", "MOT20-08"]


def af(v, d=0.0):
    try:
        if v is None or v == "":
            return d
        return float(v)
    except Exception:
        return d


def ai(v, d=0):
    try:
        if v is None or v == "":
            return d
        return int(float(v))
    except Exception:
        return d


def read_csv(path: Path) -> List[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    fields: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fields.append(k)
    if not fields:
        fields = ["seq", "track_a", "track_b"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def read_mot(path: Path) -> List[Tuple[int, int, List[str]]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 6:
                continue
            rows.append((int(float(parts[0])), int(float(parts[1])), parts))
    return rows


def line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find(parent: Dict[int, int], x: int) -> int:
    parent.setdefault(x, x)
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def source_file(source_by_seq_dir: Path, seq: str) -> Path:
    p1 = source_by_seq_dir / seq / f"{seq}.txt"
    if p1.exists():
        return p1
    p2 = source_by_seq_dir / f"{seq}.txt"
    if p2.exists():
        return p2
    raise FileNotFoundError(f"missing source MOT file for {seq} under {source_by_seq_dir}")


def gate_pass(row: dict, cfg: dict) -> bool:
    if af(row.get("aflink_score")) < float(cfg.get("min_aflink", -1.0)):
        return False
    if "min_adjusted" in cfg and af(row.get("debt_adjusted_edge_score")) < float(cfg["min_adjusted"]):
        return False
    if af(row.get("out_rank_by_aflink_score"), 999) > float(cfg.get("max_out_rank", 999)):
        return False
    if af(row.get("in_rank_by_aflink_score"), 999) > float(cfg.get("max_in_rank", 999)):
        return False
    if min(af(row.get("out_margin_to_second_aflink_score")), af(row.get("in_margin_to_second_aflink_score"))) < float(cfg.get("min_bidir_margin", -999)):
        return False
    if af(row.get("edge_debt_score")) < float(cfg.get("min_debt", 0)):
        return False
    if af(row.get("risk_total")) > float(cfg.get("max_risk", 999)):
        return False
    if af(row.get("geometry_risk")) > float(cfg.get("max_geometry_risk", 999)):
        return False
    if af(row.get("motion_risk")) > float(cfg.get("max_motion_risk", 999)):
        return False
    if af(row.get("competition_risk")) > float(cfg.get("max_competition_risk", 999)):
        return False
    if "min_appearance_max" in cfg and af(row.get("appearance_max")) < float(cfg["min_appearance_max"]):
        return False
    es = cfg.get("edge_set")
    if es == "no_highrisk" and row.get("edge_type") == "high_risk_geometry":
        return False
    if es == "boundary_or_fragment" and row.get("edge_type") not in {"weak_boundary_recovery", "fragmented_tracklet_recovery"}:
        return False
    if es == "stable_gap" and row.get("edge_type") not in {"short_gap_continuation", "long_gap_reappearance"}:
        return False
    gap = ai(row.get("gap"), -1)
    if gap <= 0:
        return False
    return True


def package_zip(package_root: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for seq in MOT20_TEST_SEQS:
            f = package_root / f"{seq}.txt"
            if not f.exists():
                raise FileNotFoundError(f"missing package file {f}")
            zf.write(f, arcname=f.name)


def run_validation(result_dir: Path, zip_path: Path, out_dir: Path) -> dict:
    logs = {}
    cmds = {
        "result_dir": [sys.executable, "scripts/check_mot20_submission.py", "--results-dir", str(result_dir), "--profile", "mot20_test_4"],
        "zip_path": [sys.executable, "scripts/check_mot20_submission.py", "--zip-path", str(zip_path), "--profile", "mot20_test_4"],
    }
    for name, cmd in cmds.items():
        p = subprocess.run(cmd, cwd=".", text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        log_path = out_dir / f"validation_{name}.txt"
        log_path.write_text(p.stdout, encoding="utf-8")
        logs[name] = {"returncode": p.returncode, "log_path": str(log_path)}
    return logs


def solve_policy(policy_name: str, cfg: dict, candidates: List[dict], source_by_seq_dir: Path, out_root: Path) -> dict:
    policy_dir = out_root / policy_name
    linked_dir = policy_dir / "linked_results"
    package_root = policy_dir / "package_root"
    linked_dir.mkdir(parents=True, exist_ok=True)
    package_root.mkdir(parents=True, exist_ok=True)

    gated = [r for r in candidates if gate_pass(r, cfg)]
    gated_by_seq = defaultdict(list)
    for r in gated:
        gated_by_seq[r["seq"]].append(r)

    selected_all: List[dict] = []
    by_seq = []
    input_audit = []

    for seq in MOT20_TEST_SEQS:
        seq_candidates = gated_by_seq.get(seq, [])
        G = nx.Graph()
        payload = {}
        for i, r in enumerate(seq_candidates):
            a = ai(r.get("track_a"))
            b = ai(r.get("track_b"))
            w = af(r.get("debt_adjusted_edge_score"))
            # Use aflink and deterministic index only for stable tie-breaking.
            w = w + 1e-9 * af(r.get("aflink_score")) - 1e-12 * i
            if w <= 0:
                continue
            u = ("src", a)
            v = ("dst", b)
            G.add_edge(u, v, weight=float(w))
            payload[(u, v)] = r
            payload[(v, u)] = r

        matching = nx.algorithms.matching.max_weight_matching(G, maxcardinality=False, weight="weight")
        matched = []
        for u, v in matching:
            r = payload.get((u, v))
            if r is not None:
                matched.append(r)
        matched.sort(key=lambda r: (af(r.get("debt_adjusted_edge_score")), af(r.get("aflink_score"))), reverse=True)

        parent: Dict[int, int] = {}
        used_successor = set()
        used_predecessor = set()
        final_selected = []
        for r in matched:
            a = ai(r.get("track_a"))
            b = ai(r.get("track_b"))
            if a in used_successor or b in used_predecessor:
                continue
            ra, rb = find(parent, a), find(parent, b)
            if ra == rb:
                continue
            parent[rb] = ra
            used_successor.add(a)
            used_predecessor.add(b)
            q = dict(r)
            q["policy"] = policy_name
            q["selected_rank_in_seq"] = len(final_selected) + 1
            q["edge_weight"] = af(r.get("debt_adjusted_edge_score"))
            final_selected.append(q)

        involved = set()
        for r in final_selected:
            involved.add(ai(r.get("track_a")))
            involved.add(ai(r.get("track_b")))
        id_map = {tid: find(parent, tid) for tid in involved}

        src = source_file(source_by_seq_dir, seq)
        out_file = linked_dir / f"{seq}.txt"
        rows = []
        for _, tid, parts in read_mot(src):
            p = list(parts)
            p[1] = str(id_map.get(tid, tid))
            rows.append(p)
        rows.sort(key=lambda p: (int(float(p[0])), int(float(p[1])), float(p[2]), float(p[3])))
        with out_file.open("w", encoding="utf-8") as f:
            for p in rows:
                f.write(",".join(p) + "\n")
        shutil.copy2(out_file, package_root / out_file.name)

        selected_all.extend(final_selected)
        by_seq.append({
            "seq": seq,
            "gated_candidates": len(seq_candidates),
            "graph_edges": G.number_of_edges(),
            "matching_edges": len(matched),
            "accepted_links": len(final_selected),
            "mean_aflink": sum(af(r.get("aflink_score")) for r in final_selected) / len(final_selected) if final_selected else 0.0,
            "mean_debt_adjusted": sum(af(r.get("debt_adjusted_edge_score")) for r in final_selected) / len(final_selected) if final_selected else 0.0,
        })
        input_audit.append({
            "seq": seq,
            "source_file": str(src),
            "source_rows": line_count(src),
            "source_md5": md5(src),
            "linked_file": str(out_file),
            "linked_rows": line_count(out_file),
            "linked_md5": md5(out_file),
            "row_count_ok": int(line_count(src) == line_count(out_file)),
        })

    write_csv(policy_dir / "accepted_links.csv", selected_all)
    write_csv(policy_dir / "input_output_audit.csv", input_audit)

    zip_path = policy_dir / f"MOT20_A41_{policy_name}_submission.zip"
    package_zip(package_root, zip_path)
    validation = run_validation(package_root, zip_path, policy_dir)

    summary = {
        "policy": policy_name,
        "gate_config": cfg,
        "source_by_seq_dir": str(source_by_seq_dir),
        "candidate_edges_total": len(candidates),
        "gated_candidates_total": len(gated),
        "accepted_links_total": len(selected_all),
        "by_seq": by_seq,
        "input_output_audit": input_audit,
        "zip_path": str(zip_path),
        "validation": validation,
        "decision": "PASS_FORMAT_READY" if all(x["row_count_ok"] for x in input_audit) and all(v["returncode"] == 0 for v in validation.values()) else "CHECK_FAILED",
    }
    (policy_dir / "link_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--gate-configs", required=True)
    ap.add_argument("--source-by-seq-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--policies", nargs="*", default=["strict_p80", "balanced_p70", "aggressive_p60"])
    args = ap.parse_args()

    candidates = read_csv(Path(args.candidates))
    gate_configs = json.loads(Path(args.gate_configs).read_text(encoding="utf-8"))
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    source_dir = Path(args.source_by_seq_dir)

    summaries = []
    for policy in args.policies:
        if policy not in gate_configs:
            raise KeyError(f"policy {policy} not found in {args.gate_configs}")
        summaries.append(solve_policy(policy, gate_configs[policy], candidates, source_dir, out_root))

    aggregate = {
        "candidates": str(args.candidates),
        "gate_configs": str(args.gate_configs),
        "source_by_seq_dir": str(source_dir),
        "policies": summaries,
        "decision": "A41_02_SOLVER_V1_DONE",
    }
    (out_root / "a41_02_summary.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md = ["# A41_02 Global Solver v1", "", "```json", json.dumps(aggregate, indent=2, sort_keys=True)[:12000], "```", ""]
    (out_root / "decision.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({
        "out_dir": str(out_root),
        "policies": [{"policy": s["policy"], "gated": s["gated_candidates_total"], "accepted": s["accepted_links_total"], "decision": s["decision"], "zip": s["zip_path"]} for s in summaries],
        "decision": aggregate["decision"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

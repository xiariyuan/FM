#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
import run_a39_rule_stress as stress

REPO = Path(__file__).resolve().parents[1]


def read_csv(path: Path):
    if not path.exists():
        return []
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows):
    fields=[]; seen=set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k); fields.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=fields or ['anchor_id'], extrasaction='ignore')
        w.writeheader(); w.writerows(rows)


def ai(v,d=0):
    try:return int(float(v))
    except Exception:return d


def run_path_builder(c, run_dir: Path, timeout_s: int, device: str):
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd=[
        sys.executable, 'scripts/simulate_a39_path_bridge_rewrite.py',
        '--track-file','outputs/spot_runtime_gate_20260628/A37_identity_escrow_association/A37_00b_candidate_logger_seq02/track_results/MOT20-02.txt',
        '--gt-file','datasets/MOT20/train/MOT20-02/gt/gt.txt',
        '--tunnels-csv','outputs/spot_runtime_gate_20260628/A39_identity_suspension_tunnel/A39_00_tunnel_discovery_seq02_ioa050_dur3/tunnel_candidates.csv',
        '--img-dir','datasets/MOT20/train/MOT20-02/img1',
        '--fast-reid-config','external/BoT-SORT-main/fast_reid/configs/MOT20/sbs_S50.yml',
        '--fast-reid-weights','external/BoT-SORT-main/pretrained/mot20_sbs_S50.pth',
        '--out-dir',str(run_dir),
        '--tunnel-id',str(ai(c.get('tunnel_id'))),
        '--pre-anchor',str(ai(c.get('pre_track'))),
        '--post-anchor',str(ai(c.get('post_track'))),
        '--target-id',str(ai(c.get('pre_track'))),
        '--anchor-gt',str(ai(c.get('pre_gt',-1),-1)),
        '--device',device,
    ]
    with (run_dir/'path_builder_stdout.log').open('w') as out, (run_dir/'path_builder_stderr.log').open('w') as err:
        try:
            ret=subprocess.run(cmd,cwd=REPO,stdout=out,stderr=err,text=True,timeout=timeout_s)
            return 'ok' if ret.returncode==0 else f'path_builder_failed_{ret.returncode}'
        except subprocess.TimeoutExpired:
            return 'timeout_killed'


def placeholder(c, out: Path, status: str):
    aid=c.get('anchor_id') or stress.anchor_id(c.get('tunnel_id'),c.get('pre_track'),c.get('post_track'))
    return {
        'anchor_id':aid,'candidate_source':c.get('candidate_source',''),'run_status':status,
        'source_run_dir':str(out/'micro'/f'anchor_{aid}'),
        'tunnel_id':c.get('tunnel_id',''),'pre_track':c.get('pre_track',''),'post_track':c.get('post_track',''),
        'pre_gt':c.get('pre_gt',''),'post_gt':c.get('post_gt',''),'gt_same':c.get('gt_same',''),
        'sim':c.get('sim',''),'row_rank':c.get('row_rank',''),'col_rank':c.get('col_rank',''),
        'feature_lifecycle_suspension':c.get('feature_lifecycle_suspension',''),'feature_top11':c.get('feature_top11',''),
        'label_safe_to_rewrite':'0','reject_reason':status,
        'rule_current':'0','rule_no_bridge':'0','rule_direct_mode':'0','rule_bridge_mode':'0','rule_mode_union':'0'
    }


def finalize(out: Path, rows):
    write_csv(out/'path_builder_summary.csv', rows)
    write_csv(out/'path_transaction_examples_stress.csv', rows)
    rule_defs=[('rule_current',stress.rule_current),('rule_no_bridge',stress.rule_no_bridge),('rule_direct_mode',stress.rule_direct_mode),('rule_bridge_mode',stress.rule_bridge_mode),('rule_mode_union',stress.rule_mode_union)]
    rule_rows=[]; fp_cases=[]
    for name,fn in rule_defs:
        rep,fps=stress.summarize_rule(rows,name,fn); rule_rows.append(rep)
        for f in fps:
            rec=dict(f); rec['rule_name']=name; rec['why_dangerous']=rec.get('reject_reason','false_positive'); fp_cases.append(rec)
    safe_cases=[r for r in rows if ai(r.get('label_safe_to_rewrite'))==1]
    write_csv(out/'rule_stress_report.csv', rule_rows)
    write_csv(out/'false_positive_cases.csv', fp_cases)
    write_csv(out/'safe_path_cases.csv', safe_cases)
    summary={'candidate_count': len(read_csv(out/'candidate_anchor_manifest.csv')), 'path_examples':len(rows), 'safe_path_examples':len(safe_cases), 'unsafe_path_examples':len(rows)-len(safe_cases), 'rule_report':rule_rows, 'false_positive_cases':len(fp_cases)}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    md=['# A39_04c Candidate Expansion false TopN and Rule Stress','','## Summary','','```json',json.dumps(summary,indent=2,sort_keys=True),'```','','## Rule stress report','','| rule | tp | fp | tn | fn | precision | recall | accepted | safe |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rule_rows:
        md.append(f"| {r['rule_name']} | {r['tp']} | {r['fp']} | {r['tn']} | {r['fn']} | {r['precision']:.4f} | {r['recall']:.4f} | {r['accepted_count']} | {r['safe_count']} |")
    md += ['','## Safe path cases','','| anchor | source | sim | planned | wrong | skip | HOTA | IDF1 | IDSW |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in safe_cases:
        md.append(f"| {r.get('anchor_id')} | {r.get('candidate_source')} | {r.get('sim')} | {r.get('planned_rows')} | {r.get('diag_wrong_rows')} | {r.get('skipped_collision_rows')} | {r.get('HOTA')} | {r.get('IDF1')} | {r.get('IDSW')} |")
    md += ['','## False positives','']
    if fp_cases:
        md += ['| rule | anchor | source | sim | planned | wrong | skip | reason |','|---|---|---|---:|---:|---:|---:|---|']
        for r in fp_cases:
            md.append(f"| {r.get('rule_name')} | {r.get('anchor_id')} | {r.get('candidate_source')} | {r.get('sim')} | {r.get('planned_rows')} | {r.get('diag_wrong_rows')} | {r.get('skipped_collision_rows')} | {r.get('reject_reason')} |")
    else:
        md.append('No false positives in this stress set.')
    (out/'decision.md').write_text('\n'.join(md)+'\n')
    (out/'data_gap_report.md').write_text('\n'.join(md)+'\n')
    print('\n'.join(md))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out-dir',default='outputs/spot_runtime_gate_20260628/A39_identity_suspension_tunnel/A39_04c_candidate_expansion_false_topN_and_rule_stress')
    ap.add_argument('--a39-root',default='outputs/spot_runtime_gate_20260628/A39_identity_suspension_tunnel')
    ap.add_argument('--a39-04b-root',default='outputs/spot_runtime_gate_20260628/A39_identity_suspension_tunnel/A39_04b_rule_baseline_and_path_expansion')
    ap.add_argument('--timeout-s',type=int,default=90)
    ap.add_argument('--device',default='cuda')
    args=ap.parse_args()
    out=Path(args.out_dir); a39=Path(args.a39_root); b=Path(args.a39_04b_root)
    candidates=read_csv(out/'candidate_anchor_manifest.csv')
    rows=read_csv(out/'path_builder_summary.csv')
    seen={r['anchor_id'] for r in rows}
    for c in candidates:
        aid=c['anchor_id']
        if aid in seen:
            continue
        run_dir=stress.known_run(aid,a39,b) or (out/'micro'/f'anchor_{aid}')
        status='reused_existing' if (run_dir/'rewrite_summary.json').exists() else run_path_builder(c,run_dir,args.timeout_s,args.device)
        if (run_dir/'rewrite_summary.json').exists():
            row=stress.summarize_run(c,run_dir,status,c.get('candidate_source',''),True)
        else:
            row=placeholder(c,out,status)
        rows.append(row); seen.add(aid)
        write_csv(out/'path_builder_summary.csv', rows)
        print('done',aid,status)
    finalize(out,rows)

if __name__=='__main__':
    main()

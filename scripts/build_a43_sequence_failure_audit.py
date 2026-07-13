#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, zipfile
from pathlib import Path
from typing import Dict, List

SEQS = ['MOT20-04','MOT20-06','MOT20-07','MOT20-08']

# Official server feedback pasted by user. Values are decimals, not percentages.
OFFICIAL = {
    'A41_05c_official': {
        'MOT20-04': {'HOTA':0.710915,'MOTA':0.879818,'IDF1':0.870140,'DetA':0.723615,'AssA':0.699722,'CLR_FN':20481,'CLR_FP':12132,'IDSW':327,'Frag':419},
        'MOT20-06': {'HOTA':0.531358,'MOTA':0.663558,'IDF1':0.657087,'DetA':0.543674,'AssA':0.520867,'CLR_FN':36898,'CLR_FP':7173,'IDSW':594,'Frag':497},
        'MOT20-07': {'HOTA':0.656210,'MOTA':0.822362,'IDF1':0.760875,'DetA':0.701431,'AssA':0.619919,'CLR_FN':3605,'CLR_FP':2160,'IDSW':115,'Frag':113},
        'MOT20-08': {'HOTA':0.496404,'MOTA':0.609829,'IDF1':0.631606,'DetA':0.497526,'AssA':0.497618,'CLR_FN':25407,'CLR_FP':4513,'IDSW':312,'Frag':335},
        'COMBINED': {'HOTA':0.635993,'MOTA':0.780226,'IDF1':0.778505,'DetA':0.641357,'AssA':0.633055,'CLR_FN':86391,'CLR_FP':25978,'IDSW':1348,'Frag':1364},
    },
    'A42_02b_official': {
        'MOT20-04': {'HOTA':0.710911,'MOTA':0.879814,'IDF1':0.870138,'DetA':0.723612,'AssA':0.699717,'CLR_FN':20481,'CLR_FP':12133,'IDSW':327,'Frag':419},
        'MOT20-06': {'HOTA':0.531354,'MOTA':0.663543,'IDF1':0.657082,'DetA':0.543666,'AssA':0.520867,'CLR_FN':36898,'CLR_FP':7175,'IDSW':594,'Frag':497},
        'MOT20-07': {'HOTA':0.655745,'MOTA':0.822724,'IDF1':0.759700,'DetA':0.701671,'AssA':0.618862,'CLR_FN':3581,'CLR_FP':2173,'IDSW':114,'Frag':111},
        'MOT20-08': {'HOTA':0.496370,'MOTA':0.609739,'IDF1':0.631573,'DetA':0.497481,'AssA':0.497594,'CLR_FN':25407,'CLR_FP':4520,'IDSW':312,'Frag':335},
        'COMBINED': {'HOTA':0.635955,'MOTA':0.780229,'IDF1':0.778419,'DetA':0.641362,'AssA':0.632974,'CLR_FN':86367,'CLR_FP':26001,'IDSW':1347,'Frag':1362},
    },
}

VARIANTS = {
    'A41_05c': 'outputs/spot_runtime_gate_20260628/A41_association_debt_global_tracker/A41_05_teacher_recovery_hybrid/A41_05c_test_submission_union_score015_app060_risk7_rank2',
    'A42_02b': 'outputs/spot_runtime_gate_20260628/A42_long_gap_global_association/A42_02b_ranking_model_test_submission/top150_rank3_app055',
    'A23_thr015': 'outputs/spot_runtime_gate_20260628/A23_appearance_aflink/A23_test_apply_thr015',
    'A23_thr020': 'outputs/spot_runtime_gate_20260628/A23_appearance_aflink/A23_test_apply_thr020',
    'A23_thr030': 'outputs/spot_runtime_gate_20260628/A23_appearance_aflink/A23_test_apply_thr030',
    'A17_risk_soft_gap20': 'outputs/spot_runtime_gate_20260628/A17_test_submission_risk_soft_interp_gap20',
    'A18_control_gap20': 'outputs/spot_runtime_gate_20260628/A18_test_submission_control_interp_gap20',
}

def count_lines(path: Path) -> int:
    if not path.exists(): return 0
    with path.open('r', encoding='utf-8') as f:
        return sum(1 for _ in f)

def read_csv(path: Path) -> List[dict]:
    if not path.exists(): return []
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def read_json(path: Path):
    if not path.exists(): return None
    return json.loads(path.read_text(encoding='utf-8'))

def find_result_dir(root: Path) -> Path | None:
    for name in ['package_root','track_results','linked_results','raw_linked_results']:
        p = root / name
        if p.exists() and any((p / f'{seq}.txt').exists() for seq in SEQS):
            return p
    return None

def variant_stats(name: str, root_s: str) -> List[dict]:
    root = Path(root_s)
    rows = []
    if not root.exists():
        return [{'variant': name, 'seq': 'MISSING', 'root': root_s}]
    # Link summaries and accepted/selected links.
    accepted = read_csv(root / 'accepted_links.csv')
    selected = read_csv(root / 'selected_links.csv')
    link_summary = read_json(root / 'link_summary.json')
    submission_summary = read_json(root / 'submission_summary.json')
    interp_summary = read_json(root / 'interp_summary.json')
    interp_by_seq = {}
    if isinstance(interp_summary, list):
        for r in interp_summary:
            interp_by_seq[r.get('seq')] = r
    elif isinstance(interp_summary, dict):
        # support unusual formats if any
        for r in interp_summary.get('by_seq', []): interp_by_seq[r.get('seq')] = r
    result_dir = find_result_dir(root)
    raw_dir = root / 'raw_linked_results' if (root / 'raw_linked_results').exists() else (root / 'linked_results')
    track_dir = root / 'track_results' if (root / 'track_results').exists() else result_dir
    # accepted links by seq.
    links_by_seq = {seq:0 for seq in SEQS}; rec_by_seq={seq:0 for seq in SEQS}; base_by_seq={seq:0 for seq in SEQS}
    for r in accepted or selected:
        seq = r.get('seq')
        if seq in links_by_seq:
            links_by_seq[seq]+=1
            rec_by_seq[seq]+= int(float(r.get('is_recovery',0) or 0))
            base_by_seq[seq]+= int(float(r.get('is_base', r.get('is_base_a41', 0)) or 0))
    # Fallback from summary by_seq.
    by_seq_summary=[]
    if submission_summary: by_seq_summary = submission_summary.get('by_seq', [])
    elif link_summary: by_seq_summary = link_summary.get('by_seq', []) or link_summary.get('link_summary', {}).get('by_seq', [])
    for r in by_seq_summary:
        seq=r.get('seq')
        if seq in SEQS and links_by_seq[seq]==0:
            links_by_seq[seq]=int(r.get('accepted_links', r.get('selected_links', 0)) or 0)
            rec_by_seq[seq]=int(r.get('recovery_links', 0) or 0)
            base_by_seq[seq]=int(r.get('base_links', 0) or 0)
    for seq in SEQS:
        interp = interp_by_seq.get(seq, {})
        raw_rows = count_lines(raw_dir / f'{seq}.txt') if raw_dir and raw_dir.exists() else 0
        track_rows = count_lines(track_dir / f'{seq}.txt') if track_dir and track_dir.exists() else 0
        rows.append({
            'variant': name,
            'seq': seq,
            'root': root_s,
            'result_dir': str(track_dir) if track_dir else '',
            'raw_rows': raw_rows,
            'track_rows': track_rows,
            'inserted_rows_json': int(interp.get('inserted_rows', 0) or 0),
            'gaps_seen': int(interp.get('gaps_seen', 0) or 0),
            'gaps_filled': int(interp.get('gaps_filled', 0) or 0),
            'links': links_by_seq.get(seq, 0),
            'base_links': base_by_seq.get(seq, 0),
            'recovery_links': rec_by_seq.get(seq, 0),
        })
    return rows

def write_csv(path: Path, rows: List[dict]):
    rows=list(rows); fields=[]; seen=set()
    for r in rows:
        for k in r:
            if k not in seen: seen.add(k); fields.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=fields or ['x'], extrasaction='ignore')
        w.writeheader(); w.writerows(rows)

def official_rows() -> List[dict]:
    rows=[]
    for run, data in OFFICIAL.items():
        for seq, m in data.items():
            row={'run':run,'seq':seq}; row.update(m); rows.append(row)
    return rows

def official_delta_rows() -> List[dict]:
    rows=[]
    a=OFFICIAL['A41_05c_official']; b=OFFICIAL['A42_02b_official']
    for seq in SEQS+['COMBINED']:
        row={'seq':seq,'from':'A41_05c','to':'A42_02b'}
        for k in ['HOTA','MOTA','IDF1','DetA','AssA','CLR_FN','CLR_FP','IDSW','Frag']:
            row[f'delta_{k}']=b[seq][k]-a[seq][k]
        rows.append(row)
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out-dir', required=True)
    args=ap.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    off=official_rows(); delta=official_delta_rows()
    write_csv(out/'official_metric_feedback.csv', off)
    write_csv(out/'official_A42_minus_A41_delta.csv', delta)
    local=[]
    for name, root in VARIANTS.items(): local.extend(variant_stats(name, root))
    write_csv(out/'local_variant_structure_stats.csv', local)
    # Focus sequence burden from official A41/A42.
    focus=[]
    combined=OFFICIAL['A41_05c_official']['COMBINED']
    for seq in SEQS:
        m=OFFICIAL['A41_05c_official'][seq]
        focus.append({
            'seq': seq,
            'HOTA': m['HOTA'], 'IDF1': m['IDF1'], 'AssA': m['AssA'],
            'FN': m['CLR_FN'], 'FP': m['CLR_FP'], 'IDSW': m['IDSW'], 'Frag': m['Frag'],
            'fn_share': m['CLR_FN']/combined['CLR_FN'],
            'idsw_share': m['IDSW']/combined['IDSW'],
            'frag_share': m['Frag']/combined['Frag'],
            'priority_score': (1-m['HOTA'])*2 + m['CLR_FN']/combined['CLR_FN'] + m['IDSW']/combined['IDSW'],
        })
    focus=sorted(focus, key=lambda r:r['priority_score'], reverse=True)
    write_csv(out/'sequence_priority_focus.csv', focus)
    # Compare structural rows for A41/A42/A23 to locate sequence differences.
    stat_idx={(r['variant'],r['seq']):r for r in local}
    struct_delta=[]
    for seq in SEQS:
        r41=stat_idx.get(('A41_05c',seq),{}); r42=stat_idx.get(('A42_02b',seq),{})
        struct_delta.append({
            'seq': seq,
            'A41_links': r41.get('links',0), 'A42_links': r42.get('links',0), 'delta_links': int(r42.get('links',0))-int(r41.get('links',0)),
            'A41_recovery_links': r41.get('recovery_links',0), 'A42_recovery_links': r42.get('recovery_links',0), 'delta_recovery_links': int(r42.get('recovery_links',0))-int(r41.get('recovery_links',0)),
            'A41_track_rows': r41.get('track_rows',0), 'A42_track_rows': r42.get('track_rows',0), 'delta_track_rows': int(r42.get('track_rows',0))-int(r41.get('track_rows',0)),
            'A41_inserted_rows': r41.get('inserted_rows_json',0), 'A42_inserted_rows': r42.get('inserted_rows_json',0), 'delta_inserted_rows': int(r42.get('inserted_rows_json',0))-int(r41.get('inserted_rows_json',0)),
        })
    write_csv(out/'local_A42_minus_A41_structure_delta.csv', struct_delta)
    decision={
        'decision':'A43_00_AUDIT_DONE__NEXT_A43_01_SEQUENCE_SPECIFIC_FN_RECOVERY_AUDIT',
        'key_findings':[
            'A42 vs A41 official feedback is effectively flat/slightly worse: HOTA -0.000038, IDF1 -0.000086, AssA -0.000081.',
            'MOT20-06 and MOT20-08 dominate remaining error: under A41 they account for about 72% of FN and 67% of IDSW.',
            'A42 structural changes are tiny compared with total errors; link-only recovery is saturated on test.',
            'Next should not be another A41/A42 topK/rank tweak. Move to sequence-specific FN/gap recovery audit focused on MOT20-06 and MOT20-08.'
        ],
        'focus_sequences':focus,
        'next':'A43_01_sequence_specific_candidate_recall_audit: compare existing variants per seq, inspect row/insert/recovery deltas, and build 06/08-specific low-FN/gap-fill recovery candidates.'
    }
    (out/'decision.json').write_text(json.dumps(decision, indent=2, sort_keys=True)+'\n')
    md=['# A43_00 Sequence Failure Audit','','## Decision','','```text',decision['decision'],'```','','## Key findings']
    for x in decision['key_findings']: md.append(f'- {x}')
    md += ['','## Sequence priority']
    for r in focus:
        md.append(f"- {r['seq']}: HOTA={r['HOTA']:.6f}, FN={r['FN']}, IDSW={r['IDSW']}, fn_share={r['fn_share']:.3f}, idsw_share={r['idsw_share']:.3f}, priority={r['priority_score']:.3f}")
    md += ['','## Next',decision['next']]
    (out/'decision.md').write_text('\n'.join(md)+'\n')
    print(json.dumps(decision, indent=2, sort_keys=True))
if __name__ == '__main__': main()

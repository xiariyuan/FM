#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, hashlib, json, os, zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

TRAIN_SEQS = ['MOT20-01','MOT20-02','MOT20-03','MOT20-05']
TEST_SEQS = ['MOT20-04','MOT20-06','MOT20-07','MOT20-08']
ALL_SEQS = TRAIN_SEQS + TEST_SEQS
KEY_TRAIN_FOCUS = {'MOT20-02','MOT20-05'}
BASELINE_NAMES = {
    'A41_05_hybrid_union_score0p15_app0p60_risk7_rank2',
    'A43_01_baseline_gap30_a3_s80',
}

def fnum(x, default=0.0):
    try:
        if x is None or x == '': return default
        return float(x)
    except Exception:
        return default

def inum(x, default=0):
    try:
        if x is None or x == '': return default
        return int(float(x))
    except Exception:
        return default

def rel(p: Path) -> str:
    try: return str(p.relative_to(Path.cwd()))
    except Exception: return str(p)

def read_csv(path: Path) -> List[dict]:
    try:
        with path.open(newline='', encoding='utf-8') as f:
            return list(csv.DictReader(f))
    except UnicodeDecodeError:
        with path.open(newline='', encoding='latin-1') as f:
            return list(csv.DictReader(f))
    except Exception:
        return []

def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    fields=[]; seen=set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k); fields.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=fields or ['x'], extrasaction='ignore')
        w.writeheader(); w.writerows(rows)

def pct_value(row: dict, *names: str) -> float:
    for name in names:
        if name in row and row.get(name) not in (None, ''):
            v=fnum(row.get(name))
            # detailed AUC fields are 0..1; summary fields are normally 0..100.
            if abs(v) <= 1.5:
                return v * 100.0
            return v
    return 0.0

def raw_value(row: dict, *names: str) -> float:
    for name in names:
        if name in row and row.get(name) not in (None, ''):
            return fnum(row.get(name))
    return 0.0

def infer_tracker_name(summary_or_detailed: Path) -> str:
    # Typical: .../eval/<tracker_name>/pedestrian_summary.txt
    if summary_or_detailed.parent.name.startswith('A') or summary_or_detailed.parent.name:
        return summary_or_detailed.parent.name
    return summary_or_detailed.parent.parent.name

def parse_detailed(path: Path) -> List[dict]:
    tracker = infer_tracker_name(path)
    rows=[]
    for r in read_csv(path):
        seq = r.get('seq') or r.get('Sequence') or ''
        if not seq: continue
        out = {
            'tracker': tracker,
            'seq': seq,
            'HOTA': pct_value(r, 'HOTA___AUC', 'HOTA(0)', 'HOTA'),
            'DetA': pct_value(r, 'DetA___AUC', 'DetA(0)', 'DetA'),
            'AssA': pct_value(r, 'AssA___AUC', 'AssA(0)', 'AssA'),
            'DetRe': pct_value(r, 'DetRe___AUC', 'DetRe'),
            'DetPr': pct_value(r, 'DetPr___AUC', 'DetPr'),
            'AssRe': pct_value(r, 'AssRe___AUC', 'AssRe'),
            'AssPr': pct_value(r, 'AssPr___AUC', 'AssPr'),
            'MOTA': pct_value(r, 'MOTA'),
            'IDF1': pct_value(r, 'IDF1'),
            'CLR_Re': pct_value(r, 'CLR_Re'),
            'CLR_Pr': pct_value(r, 'CLR_Pr'),
            'CLR_FN': inum(r.get('CLR_FN')),
            'CLR_FP': inum(r.get('CLR_FP')),
            'IDSW': inum(r.get('IDSW')),
            'Frag': inum(r.get('Frag')),
            'MT': inum(r.get('MT')),
            'ML': inum(r.get('ML')),
            'CLR_Frames': inum(r.get('CLR_Frames')),
            'detailed_path': rel(path),
            'run_dir': rel(path.parents[3]) if len(path.parents) > 3 else rel(path.parent),
        }
        rows.append(out)
    return rows

def parse_summary(path: Path) -> dict:
    tracker=infer_tracker_name(path)
    lines=[x.strip() for x in path.read_text(errors='ignore').splitlines() if x.strip()]
    if len(lines)<2: return {}
    keys=lines[0].split(); vals=lines[1].split(); raw=dict(zip(keys, vals))
    out={'tracker':tracker,'seq':'COMBINED','summary_path':rel(path),'run_dir':rel(path.parents[3]) if len(path.parents)>3 else rel(path.parent)}
    for k in ['HOTA','DetA','AssA','MOTA','IDF1','CLR_Re','CLR_Pr']:
        out[k]=pct_value(raw, k)
    for k in ['CLR_FN','CLR_FP','IDSW','Frag','MT','ML','CLR_Frames']:
        out[k]=inum(raw.get(k))
    return out

def file_head_hash(path: Path, max_bytes: int = 1_000_000) -> str:
    h=hashlib.sha1()
    try:
        with path.open('rb') as f:
            h.update(f.read(max_bytes))
        return h.hexdigest()[:16]
    except Exception:
        return ''

def mot_file_stats(path: Path) -> dict:
    rows=0; ids=set(); frames=set(); minf=None; maxf=None
    try:
        with path.open('r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line=line.strip()
                if not line: continue
                parts=line.split(',')
                if len(parts) < 6: continue
                fr=inum(parts[0]); tid=inum(parts[1])
                rows+=1; ids.add(tid); frames.add(fr)
                minf=fr if minf is None else min(minf, fr)
                maxf=fr if maxf is None else max(maxf, fr)
    except Exception:
        pass
    return {'rows': rows, 'ids': len(ids), 'frames': len(frames), 'min_frame': minf or 0, 'max_frame': maxf or 0, 'sha1_head': file_head_hash(path)}

def candidate_result_dirs(root: Path) -> List[Path]:
    # Directories containing at least two MOT20-xx.txt files; later filtered.
    dirs=set()
    for p in root.rglob('MOT20-*.txt'):
        if 'datasets' in p.parts or '.git' in p.parts: continue
        # Ignore TrackEval copied train trackers? keep them for inventory but label later.
        dirs.add(p.parent)
    return sorted(dirs)

def inventory_result_dirs(root: Path) -> List[dict]:
    rows=[]
    for d in candidate_result_dirs(root):
        names={p.name for p in d.glob('MOT20-*.txt')}
        has_test=all(f'{s}.txt' in names for s in TEST_SEQS)
        has_train=all(f'{s}.txt' in names for s in TRAIN_SEQS)
        if not (has_test or has_train):
            continue
        seqs = TEST_SEQS if has_test else TRAIN_SEQS
        row={'result_dir':rel(d), 'has_test4':int(has_test), 'has_train4':int(has_train), 'n_mot20_txt':len(names)}
        total=0
        for seq in seqs:
            st=mot_file_stats(d/f'{seq}.txt')
            for k,v in st.items(): row[f'{seq}_{k}']=v
            total += st['rows']
        row['total_rows_selected_profile']=total
        # Heuristic labels
        s=rel(d)
        row['is_trackeval_copy']=int('/trackers/' in s or '/eval_mot20_all_train/trackers/' in s)
        row['is_submission_package_root']=int(s.endswith('package_root'))
        row['is_track_results']=int(s.endswith('track_results') or '/track_results' in s)
        rows.append(row)
    return rows

def zip_inventory(root: Path) -> List[dict]:
    rows=[]
    for z in root.rglob('*.zip'):
        if '.git' in z.parts: continue
        try:
            with zipfile.ZipFile(z) as zz:
                names=[Path(n).name for n in zz.namelist() if n.endswith('.txt')]
            test=sum(1 for s in TEST_SEQS if f'{s}.txt' in names)
            train=sum(1 for s in TRAIN_SEQS if f'{s}.txt' in names)
            rows.append({'zip_path':rel(z),'n_txt':len(names),'test_seq_count':test,'train_seq_count':train,'is_mot20_test_submission':int(test==4),'is_mot20_train_package':int(train==4),'size_bytes':z.stat().st_size})
        except Exception:
            continue
    return rows

def best_by_sequence(per_seq: List[dict], seqs: List[str]) -> List[dict]:
    rows=[]
    for seq in seqs:
        cand=[r for r in per_seq if r.get('seq')==seq and r.get('HOTA',0)>0]
        if not cand: continue
        for metric, reverse in [('HOTA', True), ('IDF1', True), ('AssA', True), ('MOTA', True), ('CLR_FN', False), ('IDSW', False), ('Frag', False)]:
            best=sorted(cand, key=lambda r:(r.get(metric, 10**18 if not reverse else -10**18)), reverse=reverse)[0]
            rows.append({'seq':seq,'best_metric':metric,'tracker':best['tracker'],'value':best.get(metric), 'HOTA':best.get('HOTA'), 'IDF1':best.get('IDF1'), 'AssA':best.get('AssA'), 'MOTA':best.get('MOTA'), 'CLR_FN':best.get('CLR_FN'), 'CLR_FP':best.get('CLR_FP'), 'IDSW':best.get('IDSW'), 'Frag':best.get('Frag'), 'detailed_path':best.get('detailed_path')})
    return rows

def tracker_map(rows: List[dict]) -> Dict[str, Dict[str, dict]]:
    mp={}
    for r in rows:
        mp.setdefault(r['tracker'], {})[r['seq']] = r
    return mp

def seq_oracle(per_seq: List[dict], baseline_tracker_hint: Optional[str] = None) -> dict:
    # Build approximate oracle by selecting best HOTA per train seq; compute unweighted avg metrics and CLEAR sums.
    selected=[]
    for seq in TRAIN_SEQS:
        cand=[r for r in per_seq if r['seq']==seq and r.get('HOTA',0)>0]
        if cand:
            selected.append(max(cand, key=lambda r:r['HOTA']))
    oracle={'selected':selected}
    if selected:
        avg_keys=['HOTA','IDF1','MOTA','DetA','AssA','DetRe','DetPr','AssRe','AssPr']
        for k in avg_keys:
            oracle[k+'_avg']=sum(fnum(r.get(k)) for r in selected)/len(selected)
        for k in ['CLR_FN','CLR_FP','IDSW','Frag']:
            oracle[k+'_sum']=sum(inum(r.get(k)) for r in selected)
    # Baseline use best matching name, fallback best known A41/A43 baseline by tracker contains.
    baseline=None
    if baseline_tracker_hint:
        baseline=[r for r in per_seq if r['tracker']==baseline_tracker_hint and r['seq'] in TRAIN_SEQS]
    if not baseline:
        baseline=[r for r in per_seq if ('A41_05_hybrid_union_score0p15_app0p60_risk7_rank2' in r['tracker'] or 'A43_01_baseline_gap30_a3_s80' in r['tracker']) and r['seq'] in TRAIN_SEQS]
        # Prefer A43 baseline if present because it's freshly comparable.
        b43=[r for r in baseline if 'A43_01_baseline_gap30_a3_s80' in r['tracker']]
        if b43: baseline=b43
    if baseline:
        bseq={r['seq']:r for r in baseline}
        b=[bseq[s] for s in TRAIN_SEQS if s in bseq]
        if b:
            oracle['baseline_tracker']=b[0]['tracker']
            for k in ['HOTA','IDF1','MOTA','DetA','AssA','DetRe','DetPr','AssRe','AssPr']:
                oracle['baseline_'+k+'_avg']=sum(fnum(r.get(k)) for r in b)/len(b)
                oracle['delta_'+k+'_avg']=oracle.get(k+'_avg',0)-oracle['baseline_'+k+'_avg']
            for k in ['CLR_FN','CLR_FP','IDSW','Frag']:
                oracle['baseline_'+k+'_sum']=sum(inum(r.get(k)) for r in b)
                oracle['delta_'+k+'_sum']=oracle.get(k+'_sum',0)-oracle['baseline_'+k+'_sum']
    return oracle

def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--root', default='outputs')
    ap.add_argument('--out-dir', required=True)
    args=ap.parse_args()
    root=Path(args.root)
    out=Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    detailed_paths=sorted(root.rglob('pedestrian_detailed.csv'))
    summary_paths=sorted(root.rglob('pedestrian_summary.txt'))
    per=[]
    for p in detailed_paths:
        if '.git' in p.parts: continue
        per.extend(parse_detailed(p))
    summaries=[]
    for p in summary_paths:
        if '.git' in p.parts: continue
        s=parse_summary(p)
        if s: summaries.append(s)
    write_csv(out/'per_seq_train_metrics.csv', per)
    write_csv(out/'all_trackeval_summaries.csv', summaries)
    write_csv(out/'best_by_sequence_train.csv', best_by_sequence(per, TRAIN_SEQS))

    oracle=seq_oracle(per)
    oracle_rows=[]
    for r in oracle.get('selected', []):
        oracle_rows.append({'oracle_seq':r['seq'],'selected_tracker':r['tracker'],'HOTA':r['HOTA'],'IDF1':r['IDF1'],'AssA':r['AssA'],'MOTA':r['MOTA'],'CLR_FN':r['CLR_FN'],'CLR_FP':r['CLR_FP'],'IDSW':r['IDSW'],'Frag':r['Frag'],'detailed_path':r['detailed_path']})
    write_csv(out/'sequence_oracle_selected_train.csv', oracle_rows)
    oracle_export={k:v for k,v in oracle.items() if k!='selected'}
    (out/'sequence_oracle_bound_train.json').write_text(json.dumps(oracle_export, indent=2, sort_keys=True)+'\n', encoding='utf-8')

    result_rows=inventory_result_dirs(Path('outputs/spot_runtime_gate_20260628'))
    write_csv(out/'test_result_file_inventory.csv', result_rows)
    # keep compact test-only structural stats sorted by total rows desc.
    test_rows=[r for r in result_rows if r.get('has_test4')==1]
    test_rows=sorted(test_rows, key=lambda r: inum(r.get('total_rows_selected_profile')), reverse=True)
    write_csv(out/'test_result_structure_stats.csv', test_rows)
    zips=zip_inventory(Path('outputs/spot_runtime_gate_20260628'))
    write_csv(out/'submission_zip_inventory.csv', zips)

    # Candidate test dirs that are not merely TrackEval copies.
    candidate_test=[r for r in test_rows if r.get('is_trackeval_copy')==0]
    write_csv(out/'candidate_test_result_dirs.csv', candidate_test)

    # Decision logic.
    delta_hota=oracle_export.get('delta_HOTA_avg', 0.0)
    delta_idf1=oracle_export.get('delta_IDF1_avg', 0.0)
    # Find if oracle selected multiple trackers = potential sequence-specific merge.
    sel_trackers={r['selected_tracker'] for r in oracle_rows}
    if delta_hota >= 0.5 or len(sel_trackers) > 1:
        decision='A44_00_FOUND_SEQUENCE_ORACLE_HEADROOM__NEXT_A44_01_SEQUENCE_MERGE_CANDIDATE_BUILD'
        next_step='Inspect oracle-selected trackers and map them to test result dirs/submissions; build sequence-specific candidate merge only if matching MOT20 test files exist.'
    else:
        decision='A44_00_NO_LARGE_HISTORY_POOL_ORACLE__NEXT_A45_LOW_SCORE_DETECTOR_RECALL_RECOVERY'
        next_step='Historical evaluated run pool does not show enough per-seq headroom; move to low-score detector/tracklet recall recovery for MOT20-06/08.'
    report={
        'decision':decision,
        'next':next_step,
        'n_detailed_files':len(detailed_paths),
        'n_summary_files':len(summary_paths),
        'n_per_seq_rows':len(per),
        'n_test_result_dirs':len(test_rows),
        'n_candidate_test_dirs':len(candidate_test),
        'n_submission_zips':len(zips),
        'sequence_oracle_bound_train':oracle_export,
        'oracle_selected_trackers':sorted(sel_trackers),
        'oracle_selected':oracle_rows,
    }
    (out/'decision.json').write_text(json.dumps(report, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    md=['# A44_00 Run Pool Inventory and Sequence Oracle','','## Decision','','```text',decision,'```','','## Next','',next_step,'','## Counts',f"- detailed files: {len(detailed_paths)}",f"- summary files: {len(summary_paths)}",f"- per-seq rows: {len(per)}",f"- MOT20 test result dirs: {len(test_rows)}",f"- candidate test dirs: {len(candidate_test)}",f"- submission zips: {len(zips)}",'','## Train sequence oracle']
    for k,v in oracle_export.items():
        if k.startswith('delta_') or k in {'baseline_tracker','HOTA_avg','IDF1_avg','AssA_avg','MOTA_avg','CLR_FN_sum','CLR_FP_sum','IDSW_sum','Frag_sum'}:
            md.append(f'- {k}: {v}')
    md += ['','## Oracle selected per train sequence']
    for r in oracle_rows:
        md.append(f"- {r['oracle_seq']}: {r['selected_tracker']} | HOTA={r['HOTA']:.3f}, IDF1={r['IDF1']:.3f}, AssA={r['AssA']:.3f}, FN={r['CLR_FN']}, FP={r['CLR_FP']}, IDSW={r['IDSW']}")
    (out/'decision.md').write_text('\n'.join(md)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2, sort_keys=True))

if __name__ == '__main__':
    main()

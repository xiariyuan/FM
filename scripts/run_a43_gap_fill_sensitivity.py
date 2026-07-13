#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, shutil, subprocess, sys
from pathlib import Path
from typing import Dict, List

TRAIN_SEQS = ['MOT20-01','MOT20-02','MOT20-03','MOT20-05']
FOCUS_TRAIN_SEQS = {'MOT20-02','MOT20-05'}
KEY_METRICS = ['HOTA','IDF1','MOTA','DetA','AssA','CLR_Re','CLR_Pr','CLR_FN','CLR_FP','IDSW','Frag']

POLICIES = [
    {'name':'baseline_gap30_a3_s80', 'focus_only':False, 'max_gap':30, 'max_area_ratio':3.0, 'max_center_step':80.0, 'min_endpoint_score':0.0},
    {'name':'all_gap45_a3_s80', 'focus_only':False, 'max_gap':45, 'max_area_ratio':3.0, 'max_center_step':80.0, 'min_endpoint_score':0.0},
    {'name':'all_gap60_a3_s80', 'focus_only':False, 'max_gap':60, 'max_area_ratio':3.0, 'max_center_step':80.0, 'min_endpoint_score':0.0},
    {'name':'all_gap90_a4_s100', 'focus_only':False, 'max_gap':90, 'max_area_ratio':4.0, 'max_center_step':100.0, 'min_endpoint_score':0.0},
    {'name':'focus0205_gap45_a3_s80', 'focus_only':True, 'max_gap':45, 'max_area_ratio':3.0, 'max_center_step':80.0, 'min_endpoint_score':0.0},
    {'name':'focus0205_gap60_a3_s80', 'focus_only':True, 'max_gap':60, 'max_area_ratio':3.0, 'max_center_step':80.0, 'min_endpoint_score':0.0},
    {'name':'focus0205_gap90_a4_s100', 'focus_only':True, 'max_gap':90, 'max_area_ratio':4.0, 'max_center_step':100.0, 'min_endpoint_score':0.0},
]

def run(cmd: List[str], log_path: Path | None = None) -> int:
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(p.stdout, encoding='utf-8')
    return p.returncode

def read_csv(path: Path) -> List[dict]:
    if not path.exists(): return []
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def write_csv(path: Path, rows: List[dict]) -> None:
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

def parse_summary(path: Path) -> dict:
    if not path.exists(): return {}
    lines=[x.strip() for x in path.read_text().splitlines() if x.strip()]
    if len(lines)<2: return {}
    return dict(zip(lines[0].split(), lines[1].split()))

def parse_detailed(path: Path) -> List[dict]:
    return read_csv(path)

def floatv(x, default=0.0):
    try: return float(x)
    except Exception: return default

def intv(x, default=0):
    try: return int(float(x))
    except Exception: return default

def interpolate(source: Path, out_dir: Path, pol: dict, pattern: str='*.txt') -> int:
    cmd=[sys.executable,'scripts/postprocess/linear_interpolate_mot.py',
         '--input-dir',str(source), '--output-dir',str(out_dir), '--pattern',pattern,
         '--max-gap',str(pol['max_gap']), '--max-area-ratio',str(pol['max_area_ratio']),
         '--max-center-step',str(pol['max_center_step']), '--min-endpoint-score',str(pol['min_endpoint_score']),
         '--summary-json',str(out_dir.parent/'interp_summary_full.json'), '--summary-csv',str(out_dir.parent/'interp_summary_full.csv')]
    return run(cmd, out_dir.parent/'interp_stdout.log')

def trackeval(track_dir: Path, tracker_name: str, eval_root: Path) -> dict:
    data=eval_root/'trackers'/tracker_name/'data'; data.mkdir(parents=True, exist_ok=True)
    for seq in TRAIN_SEQS:
        shutil.copy2(track_dir/f'{seq}.txt', data/f'{seq}.txt')
    seqmap=eval_root/'seqmaps'/'MOT20_train.txt'; seqmap.parent.mkdir(parents=True, exist_ok=True)
    seqmap.write_text('name\n'+'\n'.join(TRAIN_SEQS)+'\n', encoding='utf-8')
    cmd=[sys.executable,'TrackEval/scripts/run_mot_challenge.py',
         '--GT_FOLDER','datasets/MOT20/train',
         '--TRACKERS_FOLDER',str(eval_root/'trackers'),
         '--OUTPUT_FOLDER',str(eval_root/'eval'),
         '--TRACKERS_TO_EVAL',tracker_name,
         '--BENCHMARK','MOT20','--SPLIT_TO_EVAL','train','--SEQMAP_FILE',str(seqmap),
         '--SKIP_SPLIT_FOL','True','--DO_PREPROC','True','--TRACKER_SUB_FOLDER','data','--OUTPUT_SUB_FOLDER','',
         '--PRINT_ONLY_COMBINED','True','--METRICS','HOTA','CLEAR','Identity']
    rc=run(cmd, eval_root/'trackeval_stdout.log')
    summary=eval_root/'eval'/tracker_name/'pedestrian_summary.txt'
    detailed=eval_root/'eval'/tracker_name/'pedestrian_detailed.csv'
    return {'returncode':rc,'summary':parse_summary(summary),'detailed':parse_detailed(detailed),'summary_path':str(summary),'detailed_path':str(detailed)}

def materialize_policy(source: Path, base_track: Path | None, out_policy: Path, pol: dict) -> tuple[Path, int, List[dict]]:
    track_dir=out_policy/'track_results'
    # Global policy: interpolate all seq with policy.
    if not pol.get('focus_only'):
        rc=interpolate(source, track_dir, pol)
        interp_rows=read_csv(out_policy/'interp_summary_full.csv')
        return track_dir, rc, interp_rows
    # Focus policy: start from baseline gap30 output for non-focus seq, run policy on all, then keep only focus seq.
    if base_track is None:
        raise ValueError('base_track required for focus_only policy')
    tmp=out_policy/'tmp_policy_allseq'
    rc=interpolate(source, tmp, pol)
    track_dir.mkdir(parents=True, exist_ok=True)
    for seq in TRAIN_SEQS:
        src_file=(tmp if seq in FOCUS_TRAIN_SEQS else base_track)/f'{seq}.txt'
        shutil.copy2(src_file, track_dir/f'{seq}.txt')
    # Build merged interp summary: focus from tmp, non-focus from baseline summary.
    tmp_rows={r['seq']:r for r in read_csv(out_policy/'interp_summary_full.csv')}
    base_rows={r['seq']:r for r in read_csv(base_track.parent/'interp_summary_full.csv')}
    merged=[]
    for seq in TRAIN_SEQS:
        row=dict(tmp_rows[seq] if seq in FOCUS_TRAIN_SEQS else base_rows[seq])
        row['effective_policy']=pol['name'] if seq in FOCUS_TRAIN_SEQS else 'baseline_gap30_a3_s80'
        row['focus_only']=int(seq in FOCUS_TRAIN_SEQS)
        row['output_path']=str(track_dir/f'{seq}.txt')
        merged.append(row)
    write_csv(out_policy/'interp_summary_effective.csv', merged)
    return track_dir, rc, merged

def add_metric_row(rows: List[dict], policy: dict, seq: str, metrics: dict, interp: dict | None) -> None:
    r={'policy':policy['name'],'seq':seq,'focus_only':int(policy.get('focus_only',False)),
       'max_gap':policy['max_gap'],'max_area_ratio':policy['max_area_ratio'],'max_center_step':policy['max_center_step'],
       'min_endpoint_score':policy['min_endpoint_score']}
    for k in KEY_METRICS:
        r[k]=metrics.get(k,'')
    if interp:
        for k in ['input_rows','output_rows','inserted_rows','tracks','gaps_seen','gaps_filled','max_gap']:
            r[f'interp_{k}']=interp.get(k,'')
    rows.append(r)

def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--source-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    args=ap.parse_args()
    source=Path(args.source_dir)
    out=Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out/'policies.json').write_text(json.dumps(POLICIES, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    all_rows=[]; combined_rows=[]; focus_rows=[]
    baseline_track=None
    baseline_combined=None
    policy_results=[]
    for pol in POLICIES:
        pdir=out/pol['name']; pdir.mkdir(parents=True, exist_ok=True)
        track_dir, interp_rc, interp_rows=materialize_policy(source, baseline_track, pdir, pol)
        if pol['name']=='baseline_gap30_a3_s80':
            baseline_track=track_dir
        eval_res=trackeval(track_dir, 'A43_01_'+pol['name'], pdir/'eval_mot20_all_train')
        summary=eval_res['summary']; detailed=eval_res['detailed']
        if eval_res['returncode'] != 0:
            decision='EVAL_FAILED'
        else:
            decision='EVAL_DONE'
        interp_by_seq={r['seq']:r for r in interp_rows}
        for d in detailed:
            seq=d.get('seq','')
            if seq in TRAIN_SEQS or seq=='COMBINED':
                add_metric_row(all_rows, pol, seq, d, interp_by_seq.get(seq))
        if summary:
            row={'policy':pol['name'], 'seq':'COMBINED', 'eval_returncode':eval_res['returncode'], 'decision':decision,
                 'summary_path':eval_res['summary_path'], 'detailed_path':eval_res['detailed_path']}
            for k in KEY_METRICS: row[k]=summary.get(k,'')
            combined_rows.append(row)
            if pol['name']=='baseline_gap30_a3_s80': baseline_combined=summary
        # Focus seq aggregate (MOT20-02 + MOT20-05) computed from detailed sums for CLEAR/ID; HOTA/IDF1 average rough diagnostic.
        focus_det=[d for d in detailed if d.get('seq') in FOCUS_TRAIN_SEQS]
        if focus_det:
            focus_row={'policy':pol['name'], 'seq_group':'MOT20-02+MOT20-05', 'decision':decision}
            for k in ['HOTA','IDF1','MOTA','DetA','AssA']:
                focus_row[k]=sum(floatv(d.get(k)) for d in focus_det)/len(focus_det)
            for k in ['CLR_FN','CLR_FP','IDSW','Frag']:
                focus_row[k]=sum(intv(d.get(k)) for d in focus_det)
            focus_row['inserted_rows']=sum(intv(interp_by_seq.get(seq,{}).get('inserted_rows')) for seq in FOCUS_TRAIN_SEQS)
            focus_row['gaps_filled']=sum(intv(interp_by_seq.get(seq,{}).get('gaps_filled')) for seq in FOCUS_TRAIN_SEQS)
            focus_rows.append(focus_row)
        policy_results.append({'policy':pol, 'decision':decision, 'summary':summary, 'focus':focus_rows[-1] if focus_rows else {}, 'interp_returncode':interp_rc})
    write_csv(out/'a43_01_per_seq_metrics.csv', all_rows)
    write_csv(out/'a43_01_combined_metrics.csv', combined_rows)
    write_csv(out/'a43_01_focus_surrogate_metrics.csv', focus_rows)
    # Deltas vs baseline.
    base_comb=next((r for r in combined_rows if r['policy']=='baseline_gap30_a3_s80'), None)
    base_focus=next((r for r in focus_rows if r['policy']=='baseline_gap30_a3_s80'), None)
    deltas=[]
    for r in combined_rows:
        if base_comb:
            d={'policy':r['policy'],'scope':'combined'}
            for k in ['HOTA','IDF1','MOTA','DetA','AssA']:
                d['delta_'+k]=floatv(r.get(k))-floatv(base_comb.get(k))
            for k in ['CLR_FN','CLR_FP','IDSW','Frag']:
                d['delta_'+k]=intv(r.get(k))-intv(base_comb.get(k))
            deltas.append(d)
    for r in focus_rows:
        if base_focus:
            d={'policy':r['policy'],'scope':'focus0205'}
            for k in ['HOTA','IDF1','MOTA','DetA','AssA']:
                d['delta_'+k]=floatv(r.get(k))-floatv(base_focus.get(k))
            for k in ['CLR_FN','CLR_FP','IDSW','Frag','inserted_rows','gaps_filled']:
                d['delta_'+k]=intv(r.get(k))-intv(base_focus.get(k))
            deltas.append(d)
    write_csv(out/'a43_01_deltas_vs_baseline.csv', deltas)
    # Decision: prefer policy only if combined HOTA and IDF1 do not drop and focus HOTA/IDF1 improve or FN drops without IDSW/FP blowup.
    viable=[]
    for d in deltas:
        if d['scope']!='focus0205' or d['policy']=='baseline_gap30_a3_s80': continue
        comb=next((x for x in deltas if x['scope']=='combined' and x['policy']==d['policy']), {})
        if (floatv(comb.get('delta_HOTA')) >= -0.01 and floatv(comb.get('delta_IDF1')) >= -0.02 and
            (floatv(d.get('delta_HOTA')) > 0 or intv(d.get('delta_CLR_FN')) < -50) and
            intv(d.get('delta_IDSW')) <= 20 and intv(d.get('delta_CLR_FP')) <= 1000):
            viable.append({'policy':d['policy'], 'focus_delta':d, 'combined_delta':comb})
    if viable:
        best=sorted(viable, key=lambda x:(floatv(x['focus_delta'].get('delta_HOTA')), -intv(x['focus_delta'].get('delta_CLR_FN'))), reverse=True)[0]
        decision='A43_01_FOUND_VIABLE_GAP_FILL_POLICY__NEXT_TEST_SEQUENCE_SPECIFIC_SUBMISSION'
        next_step=f"Generate A43_02 test submission: apply {best['policy']} style only to MOT20-06/MOT20-08, keep MOT20-04/07 as A41_05c."
    else:
        best=None
        decision='A43_01_NO_SAFE_LONG_GAP_GAIN__NEXT_LOW_SCORE_TRACKLET_OR_DETECTOR_RECALL_AUDIT'
        next_step='Do not generate a long-gap-fill submission. Move to low-score tracklet / detector recall recovery for MOT20-06 and MOT20-08.'
    report={'decision':decision, 'best_viable':best, 'next':next_step, 'policy_results':policy_results}
    (out/'decision.json').write_text(json.dumps(report, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    md=['# A43_01 Gap Fill Sensitivity','','## Decision','','```text',decision,'```','','## Next','',next_step,'','## Deltas vs baseline','']
    for d in deltas:
        if d['scope'] in {'combined','focus0205'}:
            md.append(f"- {d['scope']} / {d['policy']}: dHOTA={floatv(d.get('delta_HOTA')):.6f}, dIDF1={floatv(d.get('delta_IDF1')):.6f}, dFN={intv(d.get('delta_CLR_FN'))}, dFP={intv(d.get('delta_CLR_FP'))}, dIDSW={intv(d.get('delta_IDSW'))}, dFrag={intv(d.get('delta_Frag'))}")
    (out/'decision.md').write_text('\n'.join(md)+'\n', encoding='utf-8')
    print(json.dumps({'decision':decision,'best_viable':best,'next':next_step}, indent=2, sort_keys=True))

if __name__=='__main__':
    main()

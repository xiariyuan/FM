#!/usr/bin/env python3
from __future__ import annotations
import csv, json, os, zipfile
from pathlib import Path

TRAIN_SEQS=['MOT20-01','MOT20-02','MOT20-03','MOT20-05']
TEST_SEQS=['MOT20-04','MOT20-06','MOT20-07','MOT20-08']
OUT=Path('outputs/spot_runtime_gate_20260628/A44_run_pool_sequence_oracle/A44_00_run_pool_inventory')
PER=OUT/'per_seq_train_metrics.csv'

ORACLE_PATTERNS=[
 'oracle','true_candidates','upper_bound','gt_','_gt','with_gt','teacher_any_all',
 'A36_error_upper_bound','A42_00_oracle','A39_identity_suspension_tunnel/A39_01_oracle',
]
TRAIN_ONLY_PATTERNS=['eval_mot20_02','halfval','val_only']
BAD_PATTERNS=['dfine_botsort_mot20_05_full_20260627_194524']  # known bad low HOTA detector plugin run

def f(v,d=0.0):
    try: return float(v)
    except: return d

def i(v,d=0):
    try: return int(float(v))
    except: return d

def read_csv(p):
    with open(p,newline='',encoding='utf-8') as fh: return list(csv.DictReader(fh))
def write_csv(p, rows):
    rows=list(rows); fields=[]; seen=set()
    for r in rows:
        for k in r:
            if k not in seen: seen.add(k); fields.append(k)
    with open(p,'w',newline='',encoding='utf-8') as fh:
        w=csv.DictWriter(fh,fieldnames=fields or ['x']); w.writeheader(); w.writerows(rows)
def is_oracle_like(r):
    s=(r.get('tracker','')+' '+r.get('detailed_path','')+' '+r.get('run_dir','')).lower()
    return any(p.lower() in s for p in ORACLE_PATTERNS)
def is_train_only(r):
    s=(r.get('tracker','')+' '+r.get('detailed_path','')+' '+r.get('run_dir','')).lower()
    return any(p.lower() in s for p in TRAIN_ONLY_PATTERNS)
def is_bad(r):
    s=(r.get('tracker','')+' '+r.get('detailed_path','')+' '+r.get('run_dir','')).lower()
    return any(p.lower() in s for p in BAD_PATTERNS)
def rel(p):
    try: return str(Path(p).relative_to(Path.cwd()))
    except: return str(p)

def best_rows(rows, seqs, name):
    out=[]
    for seq in seqs:
        cand=[r for r in rows if r.get('seq')==seq and f(r.get('HOTA'))>0]
        for metric, rev in [('HOTA',True),('IDF1',True),('AssA',True),('MOTA',True),('CLR_FN',False),('IDSW',False),('Frag',False)]:
            if not cand: continue
            b=sorted(cand,key=lambda r:f(r.get(metric),1e18 if not rev else -1e18), reverse=rev)[0]
            out.append({'pool':name,'seq':seq,'best_metric':metric,'tracker':b.get('tracker'),'value':b.get(metric),'HOTA':b.get('HOTA'),'IDF1':b.get('IDF1'),'AssA':b.get('AssA'),'MOTA':b.get('MOTA'),'CLR_FN':b.get('CLR_FN'),'CLR_FP':b.get('CLR_FP'),'IDSW':b.get('IDSW'),'Frag':b.get('Frag'),'detailed_path':b.get('detailed_path'),'run_dir':b.get('run_dir')})
    return out

def seq_oracle(rows, seqs, baseline_name_contains='A43_01_baseline_gap30_a3_s80'):
    selected=[]
    for seq in seqs:
        cand=[r for r in rows if r.get('seq')==seq and f(r.get('HOTA'))>0]
        if cand: selected.append(max(cand,key=lambda r:f(r.get('HOTA'))))
    baseline=[]
    for r in rows:
        if baseline_name_contains in (r.get('tracker','')+r.get('detailed_path','')) and r.get('seq') in seqs:
            baseline.append(r)
    base_by={r['seq']:r for r in baseline}
    base=[base_by[s] for s in seqs if s in base_by]
    def avg(key, arr): return sum(f(r.get(key)) for r in arr)/len(arr) if arr else 0
    def sm(key, arr): return sum(i(r.get(key)) for r in arr)
    rep={'selected':[{'seq':r['seq'],'tracker':r['tracker'],'HOTA':f(r['HOTA']),'IDF1':f(r['IDF1']),'AssA':f(r['AssA']),'MOTA':f(r['MOTA']),'CLR_FN':i(r['CLR_FN']),'CLR_FP':i(r['CLR_FP']),'IDSW':i(r['IDSW']),'Frag':i(r['Frag']),'detailed_path':r.get('detailed_path')} for r in selected]}
    for k in ['HOTA','IDF1','AssA','MOTA','DetA']:
        rep[k+'_avg']=avg(k, selected)
        rep['baseline_'+k+'_avg']=avg(k, base)
        rep['delta_'+k+'_avg']=rep[k+'_avg']-rep['baseline_'+k+'_avg']
    for k in ['CLR_FN','CLR_FP','IDSW','Frag']:
        rep[k+'_sum']=sm(k, selected)
        rep['baseline_'+k+'_sum']=sm(k, base)
        rep['delta_'+k+'_sum']=rep[k+'_sum']-rep['baseline_'+k+'_sum']
    rep['baseline_count']=len(base); rep['selected_count']=len(selected)
    return rep

def fast_result_inventory(root=Path('outputs/spot_runtime_gate_20260628')):
    dirs=set()
    prune_names={'.git','datasets','TrackEval','__pycache__'}
    # Search only promising result leaves to avoid huge scans.
    allowed_leaf={'track_results','package_root','linked_results','raw_linked_results','data'}
    for dirpath, dirnames, filenames in os.walk(root):
        # prune noisy dirs in-place
        dirnames[:] = [d for d in dirnames if d not in prune_names and not d.startswith('.')]
        dp=Path(dirpath)
        if dp.name not in allowed_leaf and 'submission' not in str(dp).lower() and 'test' not in str(dp).lower():
            # still descend, but only inspect files if leaf looks promising
            pass
        names=set(filenames)
        if all(f'{s}.txt' in names for s in TEST_SEQS):
            # skip TrackEval copied tracker data
            sp=str(dp)
            if '/trackers/' in sp or '/eval_mot20_all_train/' in sp:
                continue
            dirs.add(dp)
    rows=[]
    for d in sorted(dirs):
        row={'result_dir':rel(d),'leaf':d.name}
        total=0
        for seq in TEST_SEQS:
            p=d/f'{seq}.txt'
            try:
                # wc lines without parsing full columns in python
                with p.open('rb') as fh:
                    n=sum(1 for _ in fh)
            except Exception:
                n=0
            row[f'{seq}_rows']=n; total+=n
        row['total_rows']=total
        row['is_package_root']=int(d.name=='package_root')
        row['is_track_results']=int(d.name=='track_results')
        row['is_linked_results']=int(d.name in {'linked_results','raw_linked_results'})
        rows.append(row)
    rows=sorted(rows,key=lambda r:r['total_rows'], reverse=True)
    return rows

def zip_inventory(root=Path('outputs/spot_runtime_gate_20260628')):
    rows=[]
    for z in root.rglob('*.zip'):
        try:
            with zipfile.ZipFile(z) as zz:
                base=[Path(n).name for n in zz.namelist() if n.endswith('.txt')]
            test=sum(1 for s in TEST_SEQS if f'{s}.txt' in base)
            rows.append({'zip_path':rel(z),'n_txt':len(base),'test_seq_count':test,'is_mot20_test_submission':int(test==4),'size_bytes':z.stat().st_size})
        except Exception: pass
    return sorted(rows,key=lambda r:r['size_bytes'], reverse=True)

def main():
    rows=read_csv(PER)
    for r in rows:
        r['is_oracle_like']=int(is_oracle_like(r)); r['is_train_only']=int(is_train_only(r)); r['is_bad']=int(is_bad(r))
    deploy=[r for r in rows if not is_oracle_like(r) and not is_train_only(r) and not is_bad(r)]
    write_csv(OUT/'per_seq_train_metrics_classified.csv', rows)
    write_csv(OUT/'per_seq_train_metrics_deployable.csv', deploy)
    write_csv(OUT/'best_by_sequence_train_deployable.csv', best_rows(deploy, TRAIN_SEQS, 'deployable'))
    oracle_all=json.load(open(OUT/'sequence_oracle_bound_train.json')) if (OUT/'sequence_oracle_bound_train.json').exists() else {}
    oracle_deploy=seq_oracle(deploy, TRAIN_SEQS)
    (OUT/'sequence_oracle_bound_train_deployable.json').write_text(json.dumps(oracle_deploy,indent=2,sort_keys=True)+'\n')
    test_inv=fast_result_inventory()
    write_csv(OUT/'test_result_structure_stats_fast.csv', test_inv)
    zips=zip_inventory(); write_csv(OUT/'submission_zip_inventory_fast.csv', zips)
    # Test candidate dirs that look deployable and have high row coverage, but only structural, not metric verified.
    deploy_test=[]
    for r in test_inv:
        s=r['result_dir'].lower()
        oracleish=any(p.lower() in s for p in ORACLE_PATTERNS)
        if not oracleish:
            deploy_test.append(r)
    write_csv(OUT/'candidate_test_result_dirs_fast.csv', deploy_test)
    # Decision
    dH=oracle_deploy.get('delta_HOTA_avg',0)
    dI=oracle_deploy.get('delta_IDF1_avg',0)
    selected=oracle_deploy.get('selected',[])
    trackers=sorted({x['tracker'] for x in selected})
    if dH >= 0.5 and len(trackers)>1:
        decision='A44_00_DEPLOYABLE_HISTORY_POOL_HAS_SEQUENCE_HEADROOM__NEXT_A44_01_MAP_SELECTED_TRACKERS_TO_TEST_DIRS'
        next_step='Map deployable oracle-selected train trackers to MOT20 test result dirs or rerun their configs on test; then build per-seq merge candidates.'
    elif dH >= 0.2:
        decision='A44_00_SMALL_DEPLOYABLE_HEADROOM__NEXT_A44_01_OPTIONAL_SEQUENCE_MERGE_ABLATION_BUT_A45_MORE_IMPORTANT'
        next_step='A44 per-seq merge may be tried, but expected gain is limited; prioritize A45 low-score detector/tracklet recall recovery.'
    else:
        decision='A44_00_NO_DEPLOYABLE_HISTORY_HEADROOM__NEXT_A45_LOW_SCORE_DETECTOR_RECALL_RECOVERY'
        next_step='Do not spend much time on historical per-seq merge; move to recovering low-score detections/tracklets for MOT20-06/08.'
    report={'decision':decision,'next':next_step,'n_all_per_seq_rows':len(rows),'n_deployable_rows':len(deploy),'n_test_result_dirs':len(test_inv),'n_candidate_test_dirs':len(deploy_test),'n_submission_zips':len(zips),'oracle_all_gt_like':oracle_all,'oracle_deployable':oracle_deploy,'deployable_selected_trackers':trackers,'top_candidate_test_dirs':deploy_test[:20]}
    (OUT/'decision_fast.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    md=['# A44_00 Fast Completion: Deployable Pool Filter','','## Decision','','```text',decision,'```','','## Next','',next_step,'','## Deployable train oracle']
    for k,v in oracle_deploy.items():
        if k!='selected': md.append(f'- {k}: {v}')
    md += ['','## Selected deployable trackers']
    for x in selected:
        md.append(f"- {x['seq']}: {x['tracker']} | HOTA={x['HOTA']:.3f}, IDF1={x['IDF1']:.3f}, AssA={x['AssA']:.3f}, FN={x['CLR_FN']}, FP={x['CLR_FP']}, IDSW={x['IDSW']}, Frag={x['Frag']}")
    md += ['','## Important note','The larger +2.12 train HOTA oracle from the unfiltered pool is GT/oracle-like and is not directly deployable. This file filters those runs out.']
    (OUT/'decision_fast.md').write_text('\n'.join(md)+'\n')
    print(json.dumps({'decision':decision,'next':next_step,'delta_HOTA_deployable':dH,'delta_IDF1_deployable':dI,'selected_trackers':trackers,'candidate_test_dirs':len(deploy_test)},indent=2,sort_keys=True))
if __name__=='__main__': main()

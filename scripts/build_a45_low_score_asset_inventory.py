#!/usr/bin/env python3
from __future__ import annotations
import csv, json, glob
from pathlib import Path

OUT=Path('outputs/spot_runtime_gate_20260628/A45_low_score_recall_recovery/A45_00_asset_inventory')
OUT.mkdir(parents=True, exist_ok=True)

def read_json(p):
    try: return json.load(open(p))
    except Exception: return None

def write_csv(p, rows):
    rows=list(rows); fields=[]; seen=set()
    for r in rows:
        for k in r:
            if k not in seen: seen.add(k); fields.append(k)
    with open(p,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields or ['x']); w.writeheader(); w.writerows(rows)

def fv(x,d=0):
    try: return float(x)
    except: return d

def parse_trackeval_summary(path):
    lines=[x.strip() for x in Path(path).read_text().splitlines() if x.strip()]
    if len(lines)<2: return {}
    d=dict(zip(lines[0].split(), lines[1].split()))
    d['path']=path
    return d

# Detector threshold sweep rows.
sweep_rows=[]
for p in glob.glob('outputs/detector_audit/threshold_calibration_*/baseline_threshold_sweep.json'):
    rows=read_json(p) or []
    for r in rows:
        rr={'source':p, **{k:r.get(k) for k in ['score','detections','precision50','recall50','f1_50','precision75','recall75','f1_75','fp50','fn50','fp75','fn75']}}
        sweep_rows.append(rr)
write_csv(OUT/'yolox_baseline_detection_threshold_sweep.csv', sweep_rows)
# Pick useful thresholds: high f1_75 and high recall candidates.
if sweep_rows:
    best_f175=max(sweep_rows,key=lambda r:fv(r.get('f1_75')))
    best_recall75=max(sweep_rows,key=lambda r:fv(r.get('recall75')))
    low_score=sweep_rows[0]
else:
    best_f175=best_recall75=low_score={}

# Detector model assets summary.
det_assets=[]
# DFINE tracking full.
for p in glob.glob('outputs/detector_plugins/*/dfine_botsort_stats.json'):
    st=read_json(p) or {}
    ev=list(Path(p).parent.glob('eval/eval/*/pedestrian_summary.txt'))
    m=parse_trackeval_summary(str(ev[0])) if ev else {}
    det_assets.append({
        'asset':str(p),'seq':st.get('seq'),'det_rows':st.get('det_rows'),'track_rows':st.get('track_rows'),
        'score_thresh':(st.get('params') or {}).get('score_thresh'),'track_high':(st.get('params') or {}).get('track_high'),'track_low':(st.get('params') or {}).get('track_low'),
        'HOTA':m.get('HOTA'),'IDF1':m.get('IDF1'),'MOTA':m.get('MOTA'),'CLR_FN':m.get('CLR_FN'),'CLR_FP':m.get('CLR_FP'),'IDSW':m.get('IDSW'),'Frag':m.get('Frag'),
        'decision_hint':'REJECT_DFINE_AS_RECALL_SOURCE' if m and fv(m.get('HOTA')) < 50 else 'UNKNOWN_OR_SMOKE'
    })
write_csv(OUT/'detector_plugin_assets.csv', det_assets)

# Tracker calibration manifests/logs.
calib=[]
for d in sorted(Path('outputs/detector_audit').glob('tracker_calib*')):
    man=read_json(d/'run_manifest.json') or {}
    status=(d/'status.txt').read_text(errors='ignore') if (d/'status.txt').exists() else ''
    log_tail=''
    if (d/'logs/track.log').exists():
        lines=(d/'logs/track.log').read_text(errors='ignore').splitlines()
        log_tail='\n'.join(lines[-8:])
    trk=man.get('tracker',{}) or man.get('tracking',{}) or {}
    det=man.get('detector',{}) or {}
    calib.append({'run_dir':str(d),'status':status.replace('\n',' | ')[:500],'det_ckpt':det.get('ckpt'),'exp_file':det.get('exp_file'),'log_tail':log_tail.replace('\n',' | ')[:1000]})
write_csv(OUT/'tracker_calib_runs.csv', calib)

# Decision.
findings=[]
if best_f175:
    findings.append(f"Best detector F1@0.75 threshold appears around score={best_f175.get('score')} with f1_75={best_f175.get('f1_75')}, recall75={best_f175.get('recall75')}, fp75={best_f175.get('fp75')}, fn75={best_f175.get('fn75')}.")
if low_score:
    findings.append(f"Very low threshold score={low_score.get('score')} has recall50={low_score.get('recall50')} but enormous FP50={low_score.get('fp50')}; use only as controlled low-score second-stage, not direct detector replacement.")
if det_assets:
    bad=[x for x in det_assets if x.get('decision_hint')=='REJECT_DFINE_AS_RECALL_SOURCE']
    if bad:
        findings.append('DFINE BoT-SORT MOT20-05 full run is unusable as recall source: HOTA is extremely low and FN is huge.')
findings.append('A44 deployable history-pool oracle is only +0.255 train HOTA; not enough for main breakthrough.')
findings.append('Next should run a small train-side BoT-SORT low-score/recovery threshold matrix on MOT20-02 and MOT20-05 surrogate, then apply only if HOTA/IDF1 improve without FP/IDSW explosion.')
report={
    'decision':'A45_00_ASSET_INVENTORY_DONE__NEXT_A45_01_BOTSORT_LOW_SCORE_TRACKER_THRESHOLD_MATRIX',
    'next':'Run controlled BoT-SORT/YOLOX tracker threshold matrix, not DFINE. Focus train MOT20-02/MOT20-05; candidate configs: lower track_low/new_track moderately, enable low-score association if available, keep A41/A43 postprocess afterwards.',
    'best_detector_f1_75_threshold':best_f175,
    'max_recall75_threshold':best_recall75,
    'findings':findings,
    'dfine_assets_count':len(det_assets),
    'threshold_rows':len(sweep_rows),
}
(OUT/'decision.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
md=['# A45_00 Low-score Recall Recovery Asset Inventory','','## Decision','','```text',report['decision'],'```','','## Findings']
for x in findings: md.append(f'- {x}')
md += ['','## Next',report['next'],'','## Key paths','- yolox_baseline_detection_threshold_sweep.csv','- detector_plugin_assets.csv','- tracker_calib_runs.csv']
(OUT/'decision.md').write_text('\n'.join(md)+'\n')
print(json.dumps(report,indent=2,sort_keys=True))

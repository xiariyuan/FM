#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, os, re, shutil, subprocess, sys, time
from pathlib import Path

REPO = Path('/gemini/code/FMtrack-main/FM-Track')
BOT = REPO / 'external/BoT-SORT-main'

def sh(cmd, log=None, cwd=REPO, check=True):
    print('[cmd]', ' '.join(map(str, cmd)), flush=True)
    if log:
        with open(log, 'a', encoding='utf-8') as f:
            p = subprocess.run(cmd, cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT, text=True)
    else:
        p = subprocess.run(cmd, cwd=str(cwd), text=True)
    if check and p.returncode != 0:
        raise SystemExit(p.returncode)
    return p.returncode

def parse_summary(path):
    p=Path(path)
    if not p.exists(): return {}
    lines=[l.strip() for l in p.read_text(errors='ignore').splitlines() if l.strip()]
    if len(lines)<2: return {}
    keys=lines[0].split()
    vals=lines[1].split()
    out={}
    for k,v in zip(keys, vals):
        try: out[k]=float(v)
        except: out[k]=v
    return out

def find_summary(work_dir, tracker_name):
    cands=list(Path(work_dir).glob(f'eval/{tracker_name}/pedestrian_summary.txt'))
    cands += list(Path(work_dir).glob(f'**/{tracker_name}/pedestrian_summary.txt'))
    return cands[0] if cands else None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--tracker-name', required=True)
    ap.add_argument('--seq-id', default='5')
    ap.add_argument('--baseline-summary', default='')
    ap.add_argument('--launch-next', action='store_true')
    args=ap.parse_args()
    run=Path(args.run_dir).resolve()
    trk=run/'track_results'
    trk.mkdir(exist_ok=True)
    src=BOT/'YOLOX_outputs'/args.tracker_name/'track_results'/'MOT20-05.txt'
    if not src.exists():
        print('[not_ready]', src, 'missing')
        return 2
    dst=trk/'MOT20-05.txt'
    shutil.copy2(src,dst)
    eval_dir=run/'eval'
    eval_dir.mkdir(exist_ok=True)
    log=run/'logs'/'eval.log'
    cmd=[sys.executable, str(REPO/'scripts/eval_botsort_halfval_trackeval.py'),
         '--dataset','MOT20','--data-root','/gemini/code/datasets',
         '--results-dir', str(trk), '--tracker-name', args.tracker_name,
         '--work-dir', str(eval_dir), '--remap-results-from-fullval']
    sh(cmd, log=log, cwd=REPO)
    summ=find_summary(eval_dir, args.tracker_name)
    metrics=parse_summary(summ) if summ else {}
    out={'tracker':args.tracker_name, 'summary_file':str(summ) if summ else '', 'metrics':metrics}
    if args.baseline_summary:
        base=parse_summary(args.baseline_summary)
        out['baseline_summary']=args.baseline_summary
        out['delta']={k:metrics[k]-base[k] for k in metrics if k in base and isinstance(metrics[k],float) and isinstance(base[k],float)}
    (run/'eval_summary.json').write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False), flush=True)
if __name__=='__main__':
    main()

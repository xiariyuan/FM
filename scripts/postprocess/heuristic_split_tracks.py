#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def ff(x, d=0.0):
    try: return float(x)
    except Exception: return d

def ii(x, d=0):
    try: return int(float(x))
    except Exception: return d

def center(r): return (r['x']+r['w']/2.0, r['y']+r['h']/2.0)
def area(r): return max(1e-9, r['w']*r['h'])

def read_tracks(path: Path):
    tracks=defaultdict(list)
    with path.open() as f:
        for line_no,line in enumerate(f, start=1):
            p=line.strip().split(',')
            if len(p)<6: continue
            r={'parts':p,'line_no':line_no,'frame':ii(p[0]),'tid':ii(p[1]),'x':ff(p[2]),'y':ff(p[3]),'w':ff(p[4]),'h':ff(p[5]),'score':ff(p[6],1.0) if len(p)>6 else 1.0}
            tracks[r['tid']].append(r)
    for rs in tracks.values(): rs.sort(key=lambda r:(r['frame'], r['line_no']))
    return tracks

def should_split(a,b,args):
    dt=max(1,b['frame']-a['frame'])
    ca,cb=center(a),center(b)
    center_step=math.hypot(cb[0]-ca[0], cb[1]-ca[1])/dt
    hr=max(a['h'],b['h'])/max(1e-9,min(a['h'],b['h']))
    ar=max(area(a),area(b))/max(1e-9,min(area(a),area(b)))
    score_min=min(a['score'], b['score'])
    score_drop=abs(a['score']-b['score'])
    missing=max(0,b['frame']-a['frame']-1)
    return (
        score_min <= args.score_min_le and
        score_drop >= args.score_drop_ge and
        center_step >= args.center_step_ge and
        hr >= args.height_ratio_ge and
        ar >= args.area_ratio_ge and
        missing >= args.missing_gap_ge
    )

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input-dir', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--score-min-le', type=float, default=0.8)
    ap.add_argument('--score-drop-ge', type=float, default=0.05)
    ap.add_argument('--center-step-ge', type=float, default=12.0)
    ap.add_argument('--height-ratio-ge', type=float, default=1.05)
    ap.add_argument('--area-ratio-ge', type=float, default=1.0)
    ap.add_argument('--missing-gap-ge', type=int, default=0)
    ap.add_argument('--min-seg-len', type=int, default=5)
    ap.add_argument('--summary-json', default='')
    args=ap.parse_args()
    out=Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    summary={'tracks':0,'tracks_split':0,'extra_segments':0,'split_points':0,'rows':0,'by_seq':[],'params':vars(args)}
    for txt in sorted(Path(args.input_dir).glob('MOT20-*.txt')):
        seq=txt.stem; tracks=read_tracks(txt); max_tid=max(tracks) if tracks else 0; next_tid=max_tid+1; out_rows=[]
        seq_sum={'seq':seq,'tracks':len(tracks),'tracks_split':0,'extra_segments':0,'split_points':0,'rows':0}
        for tid,rs in sorted(tracks.items()):
            seq_sum['rows']+=len(rs)
            cut_idxs=[]
            last_cut=0
            for i in range(1,len(rs)):
                if i-last_cut < args.min_seg_len: continue
                if len(rs)-i < args.min_seg_len: continue
                if should_split(rs[i-1],rs[i],args):
                    cut_idxs.append(i); last_cut=i
            cuts=[0]+cut_idxs+[len(rs)]
            if cut_idxs:
                seq_sum['tracks_split']+=1; seq_sum['extra_segments']+=len(cut_idxs); seq_sum['split_points']+=len(cut_idxs)
            for seg_i,(s,e) in enumerate(zip(cuts[:-1], cuts[1:])):
                new_tid=tid if seg_i==0 else next_tid
                if seg_i>0: next_tid+=1
                for r in rs[s:e]:
                    p=list(r['parts']); p[1]=str(new_tid); out_rows.append((r['frame'], new_tid, r['line_no'], p))
        out_rows.sort(key=lambda x:(x[0],x[1],x[2]))
        with (out/txt.name).open('w') as f:
            for _,_,_,p in out_rows: f.write(','.join(p)+'\n')
        for k in ['tracks','tracks_split','extra_segments','split_points','rows']: summary[k]+=seq_sum[k]
        summary['by_seq'].append(seq_sum)
    if args.summary_json: Path(args.summary_json).write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    print(json.dumps(summary,indent=2,sort_keys=True))
if __name__=='__main__': main()

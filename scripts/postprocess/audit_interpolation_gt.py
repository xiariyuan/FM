#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, math
from collections import defaultdict
from pathlib import Path


def load_tracks(path: Path):
    tracks=defaultdict(list)
    with path.open() as f:
        for line in f:
            line=line.strip()
            if not line: continue
            p=line.split(',')
            if len(p)<6: continue
            r=dict(frame=int(float(p[0])), tid=int(float(p[1])), x=float(p[2]), y=float(p[3]), w=float(p[4]), h=float(p[5]), score=float(p[6]) if len(p)>6 else 1.0)
            tracks[r['tid']].append(r)
    for tid in tracks: tracks[tid].sort(key=lambda r:r['frame'])
    return tracks


def load_gt(path: Path):
    by=defaultdict(list)
    with path.open() as f:
        for line in f:
            line=line.strip()
            if not line: continue
            p=line.split(',')
            if len(p)<6: continue
            fr=int(float(p[0])); gid=int(float(p[1])); x=float(p[2]); y=float(p[3]); w=float(p[4]); h=float(p[5])
            mark=int(float(p[6])) if len(p)>6 else 1
            cls=int(float(p[7])) if len(p)>7 else 1
            if mark != 1 or cls != 1: continue
            by[fr].append((gid,x,y,w,h))
    return by


def iou(a,b):
    ax,ay,aw,ah=a; bx,by,bw,bh=b
    ax2=ax+aw; ay2=ay+ah; bx2=bx+bw; by2=by+bh
    ix=max(0.0,min(ax2,bx2)-max(ax,bx)); iy=max(0.0,min(ay2,by2)-max(ay,by))
    inter=ix*iy
    if inter<=0: return 0.0
    union=max(0.0,aw)*max(0.0,ah)+max(0.0,bw)*max(0.0,bh)-inter
    return inter/union if union>0 else 0.0


def area(r): return max(0.0,r['w'])*max(0.0,r['h'])
def center(r): return (r['x']+r['w']/2.0, r['y']+r['h']/2.0)

def pass_gates(a,b,gap,args):
    if gap<=1 or gap-1>args.max_gap: return False
    if a['score']<args.min_endpoint_score or b['score']<args.min_endpoint_score: return False
    aa,bb=area(a),area(b)
    if aa<=1 or bb<=1: return False
    if max(aa,bb)/max(1e-6,min(aa,bb))>args.max_area_ratio: return False
    ca,cb=center(a),center(b)
    dist=((ca[0]-cb[0])**2+(ca[1]-cb[1])**2)**0.5
    if dist/max(1,gap)>args.max_center_step: return False
    return True

def bin_gap(g):
    if g<=5: return '02-05'
    if g<=10: return '06-10'
    if g<=15: return '11-15'
    if g<=20: return '16-20'
    if g<=25: return '21-25'
    return '26-30'

def bin_score(s):
    if s<0.5: return '<0.5'
    if s<0.6: return '0.5-0.6'
    if s<0.7: return '0.6-0.7'
    if s<0.8: return '0.7-0.8'
    return '>=0.8'

def bin_t(t):
    if t<0.25: return 'edge_0_25'
    if t<0.5: return 'mid_25_50'
    if t<0.75: return 'mid_50_75'
    return 'edge_75_100'

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--orig-dir', required=True)
    ap.add_argument('--gt-root', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--max-gap', type=int, default=30)
    ap.add_argument('--max-area-ratio', type=float, default=3.0)
    ap.add_argument('--max-center-step', type=float, default=80.0)
    ap.add_argument('--min-endpoint-score', type=float, default=0.0)
    ap.add_argument('--iou-thresh', type=float, default=0.5)
    args=ap.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rows=[]
    for mot in sorted(Path(args.orig_dir).glob('MOT20-*.txt')):
        seq=mot.stem
        tracks=load_tracks(mot)
        gt=load_gt(Path(args.gt_root)/seq/'gt'/'gt.txt')
        for tid,tr in tracks.items():
            for a,b in zip(tr,tr[1:]):
                gap=b['frame']-a['frame']
                if not pass_gates(a,b,gap,args): continue
                ca,cb=center(a),center(b)
                dist=((ca[0]-cb[0])**2+(ca[1]-cb[1])**2)**0.5
                aa,bb=area(a),area(b)
                min_score=min(a['score'],b['score'])
                for fr in range(a['frame']+1,b['frame']):
                    t=(fr-a['frame'])/gap
                    box=(a['x']+(b['x']-a['x'])*t, a['y']+(b['y']-a['y'])*t, a['w']+(b['w']-a['w'])*t, a['h']+(b['h']-a['h'])*t)
                    best=0.0; best_gid=-1
                    for gid,x,y,w,h in gt.get(fr,[]):
                        v=iou(box,(x,y,w,h))
                        if v>best: best=v; best_gid=gid
                    rows.append(dict(seq=seq, tid=tid, frame=fr, gap=gap, t=t, min_endpoint_score=min_score, area_ratio=max(aa,bb)/max(1e-6,min(aa,bb)), center_step=dist/max(1,gap), best_iou=best, best_gid=best_gid, tp=int(best>=args.iou_thresh), gap_bin=bin_gap(gap), score_bin=bin_score(min_score), t_bin=bin_t(t)))
    fields=list(rows[0].keys()) if rows else []
    with (out/'inserted_rows_audit.csv').open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    def summarize(key):
        agg=defaultdict(lambda:[0,0])
        for r in rows:
            agg[r[key]][0]+=1; agg[r[key]][1]+=int(r['tp'])
        return [{'group':k,'rows':v[0],'tp':v[1],'fp_like':v[0]-v[1],'tp_rate':v[1]/v[0] if v[0] else 0.0} for k,v in sorted(agg.items())]
    summary={
        'total_inserted': len(rows),
        'tp_like': sum(int(r['tp']) for r in rows),
        'fp_like': len(rows)-sum(int(r['tp']) for r in rows),
        'tp_rate': sum(int(r['tp']) for r in rows)/len(rows) if rows else 0.0,
        'by_seq': summarize('seq'),
        'by_gap_bin': summarize('gap_bin'),
        'by_score_bin': summarize('score_bin'),
        'by_t_bin': summarize('t_bin'),
    }
    (out/'audit_summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True)+'\n')
    md=['# Interpolation GT Audit','',f"total_inserted: {summary['total_inserted']}",f"tp_like: {summary['tp_like']}",f"fp_like: {summary['fp_like']}",f"tp_rate: {summary['tp_rate']:.4f}",'']
    for name in ['by_seq','by_gap_bin','by_score_bin','by_t_bin']:
        md += [f'## {name}','| group | rows | tp | fp_like | tp_rate |','|---|---:|---:|---:|---:|']
        for r in summary[name]: md.append(f"| {r['group']} | {r['rows']} | {r['tp']} | {r['fp_like']} | {r['tp_rate']:.4f} |")
        md.append('')
    (out/'audit_summary.md').write_text('\n'.join(md)+'\n')
    print(json.dumps(summary, indent=2, sort_keys=True))

if __name__=='__main__': main()

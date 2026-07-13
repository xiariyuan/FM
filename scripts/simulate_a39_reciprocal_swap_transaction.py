#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def ai(v, d=0):
    try:
        return int(float(v))
    except Exception:
        return d


def af(v, d=0.0):
    try:
        return float(v)
    except Exception:
        return d


def read_track(path: Path):
    rows=[]; by_frame=defaultdict(list)
    with path.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            p=line.rstrip('\n').split(',')
            if len(p)<6:
                continue
            fr=ai(p[0],-1); tid=ai(p[1],-1); x=af(p[2]); y=af(p[3]); w=af(p[4]); h=af(p[5])
            if fr<0 or tid<0 or w<=0 or h<=0:
                continue
            r={'idx':len(rows),'parts':p,'frame':fr,'track_id':tid,'box':np.array([x,y,x+w,y+h],dtype=np.float32)}
            rows.append(r); by_frame[fr].append(r)
    return rows, by_frame


def read_gt(path: Path):
    by_frame=defaultdict(list)
    with path.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            p=line.strip().split(',')
            if len(p)<6:
                continue
            fr=ai(p[0],-1); gid=ai(p[1],-1); x=af(p[2]); y=af(p[3]); w=af(p[4]); h=af(p[5])
            mark=ai(p[6],1) if len(p)>6 else 1; cls=ai(p[7],1) if len(p)>7 else 1
            if fr<0 or gid<0 or w<=0 or h<=0 or mark<=0 or cls!=1:
                continue
            by_frame[fr].append({'frame':fr,'gt_id':gid,'box':np.array([x,y,x+w,y+h],dtype=np.float32)})
    return by_frame


def pair_iou(a,b):
    if not a or not b:
        return np.zeros((len(a),len(b)),dtype=np.float32)
    A=np.stack([x['box'] for x in a]); B=np.stack([x['box'] for x in b])
    lt=np.maximum(A[:,None,:2],B[None,:,:2]); rb=np.minimum(A[:,None,2:],B[None,:,2:])
    wh=np.clip(rb-lt,0,None); inter=wh[...,0]*wh[...,1]
    aa=np.clip((A[:,2]-A[:,0])*(A[:,3]-A[:,1]),1e-6,None); bb=np.clip((B[:,2]-B[:,0])*(B[:,3]-B[:,1]),1e-6,None)
    return inter/np.clip(aa[:,None]+bb[None,:]-inter,1e-6,None)


def match_rows(rows_by_frame, gt_by_frame, iou_thr):
    row_gt={}; row_iou={}
    for fr in sorted(set(rows_by_frame)|set(gt_by_frame)):
        rr=rows_by_frame.get(fr,[]); gg=gt_by_frame.get(fr,[])
        if not rr or not gg:
            continue
        I=pair_iou(rr,gg); ri,ci=linear_sum_assignment(1-I)
        for r,c in zip(ri,ci):
            val=float(I[r,c])
            if val>=iou_thr:
                row_gt[rr[r]['idx']]=int(gg[c]['gt_id']); row_iou[rr[r]['idx']]=val
    return row_gt,row_iou


def parse_swap(s: str):
    # tunnel:track_a:track_b:start:end:gt_a:gt_b[:name]
    p=s.split(':')
    if len(p)<7:
        raise ValueError('--swap format tunnel:track_a:track_b:start:end:gt_a:gt_b[:name]')
    return {
        'tunnel_id': ai(p[0]), 'track_a': ai(p[1]), 'track_b': ai(p[2]),
        'frame_start': ai(p[3]), 'frame_end': ai(p[4]),
        'gt_a': ai(p[5]), 'gt_b': ai(p[6]),
        'swap_name': p[7] if len(p)>7 else f'{p[0]}_{p[1]}_{p[2]}_{p[3]}_{p[4]}'
    }


def should_swap(row, spec, row_gt, variant, trim):
    fr=row['frame']; tid=row['track_id']
    if tid not in {spec['track_a'], spec['track_b']}:
        return False
    fs=spec['frame_start']+trim; fe=spec['frame_end']-trim
    if fr<fs or fr>fe:
        return False
    if variant in {'full_overlap_swap','trimmed_overlap_swap'}:
        return True
    if variant=='gt_clean_swap_upper_bound':
        gid=row_gt.get(row['idx'],-1)
        if tid==spec['track_a']:
            return gid==spec['gt_b']
        if tid==spec['track_b']:
            return gid==spec['gt_a']
    raise ValueError(f'unknown variant {variant}')


def main():
    ap=argparse.ArgumentParser(description='A39_05b reciprocal two-slot swap transaction simulator.')
    ap.add_argument('--track-file', required=True)
    ap.add_argument('--gt-file', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--swap', action='append', required=True, help='tunnel:track_a:track_b:start:end:gt_a:gt_b[:name]')
    ap.add_argument('--variant', choices=['full_overlap_swap','trimmed_overlap_swap','gt_clean_swap_upper_bound'], default='full_overlap_swap')
    ap.add_argument('--trim', type=int, default=5)
    ap.add_argument('--iou-thr', type=float, default=0.5)
    args=ap.parse_args()

    out=Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rows, rows_by_frame=read_track(Path(args.track_file))
    gt_by_frame=read_gt(Path(args.gt_file))
    row_gt,row_iou=match_rows(rows_by_frame, gt_by_frame, args.iou_thr)
    specs=[parse_swap(x) for x in args.swap]

    new_parts=[r['parts'][:] for r in rows]
    audit=[]; plan=[]
    changed_indices=set()
    for spec in specs:
        planned_a=planned_b=0; correct=wrong=unknown=0
        dup_before=dup_after=0
        fs=spec['frame_start']+(args.trim if args.variant=='trimmed_overlap_swap' else 0)
        fe=spec['frame_end']-(args.trim if args.variant=='trimmed_overlap_swap' else 0)
        for r in rows:
            if not should_swap(r, spec, row_gt, args.variant, args.trim if args.variant=='trimmed_overlap_swap' else 0):
                continue
            old_id=r['track_id']
            if old_id==spec['track_a']:
                new_id=spec['track_b']; expected_gt=spec['gt_b']; planned_a+=1
            elif old_id==spec['track_b']:
                new_id=spec['track_a']; expected_gt=spec['gt_a']; planned_b+=1
            else:
                continue
            if r['idx'] in changed_indices:
                raise RuntimeError(f'row {r["idx"]} changed by multiple swaps')
            changed_indices.add(r['idx'])
            gid=row_gt.get(r['idx'],-1); giou=row_iou.get(r['idx'],0.0)
            if gid<0:
                unknown+=1; ok=0
            elif gid==expected_gt:
                correct+=1; ok=1
            else:
                wrong+=1; ok=0
            new_parts[r['idx']][1]=str(new_id)
            audit.append({
                'swap_name': spec['swap_name'], 'variant': args.variant, 'frame': r['frame'], 'idx': r['idx'],
                'old_track_id': old_id, 'new_track_id': new_id, 'row_gt': gid, 'row_gt_iou': giou,
                'expected_gt_after_swap': expected_gt, 'is_correct_after_swap': ok,
            })
        plan.append({
            **spec, 'variant': args.variant, 'effective_frame_start': fs, 'effective_frame_end': fe,
            'planned_rows_a_to_b': planned_a, 'planned_rows_b_to_a': planned_b,
            'planned_rows_total': planned_a+planned_b, 'correct_after_swap_rows': correct,
            'wrong_after_swap_rows': wrong, 'unknown_rows': unknown,
            'purity_after_swap': correct/max(1, planned_a+planned_b-unknown),
        })

    # duplicate same-frame identity after swap
    frame_id_counts=Counter()
    for p in new_parts:
        frame_id_counts[(ai(p[0]), ai(p[1]))]+=1
    dup_pairs=[(fr,tid,c) for (fr,tid),c in frame_id_counts.items() if c>1]

    track_dir=out/'track_results'; track_dir.mkdir(exist_ok=True)
    (track_dir/'MOT20-02.txt').write_text('\n'.join(','.join(p) for p in new_parts)+'\n', encoding='utf-8')
    with (out/'swap_plan.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(plan[0].keys()) if plan else ['swap_name']); w.writeheader(); w.writerows(plan)
    with (out/'swap_row_audit.csv').open('w',newline='',encoding='utf-8') as f:
        fields=list(audit[0].keys()) if audit else ['swap_name']
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(audit)
    summary={
        'variant': args.variant, 'swap_count': len(specs), 'changed_rows': len(audit),
        'correct_after_swap_rows': sum(ai(x['is_correct_after_swap']) for x in audit),
        'wrong_after_swap_rows': sum(1 for x in audit if ai(x.get('row_gt'),-1)>=0 and ai(x.get('is_correct_after_swap'))==0),
        'unknown_rows': sum(1 for x in audit if ai(x.get('row_gt'),-1)<0),
        'same_frame_duplicate_count_after_swap': len(dup_pairs),
        'duplicate_examples': dup_pairs[:20],
        'plans': plan,
    }
    (out/'gt_diagnostic_summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    (out/'gt_diagnostic_summary.md').write_text('# A39_05b swap diagnostic\n\n```json\n'+json.dumps(summary,indent=2,sort_keys=True)+'\n```\n',encoding='utf-8')
    print(json.dumps(summary,indent=2,sort_keys=True))

if __name__=='__main__':
    main()

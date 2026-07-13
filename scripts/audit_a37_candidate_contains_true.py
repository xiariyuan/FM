#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment


def parse_tlwh(s):
    try:
        vals=[float(x) for x in str(s).split(',')]
        if len(vals)!=4: return None
        x,y,w,h=vals
        return np.array([x,y,x+w,y+h], dtype=np.float32)
    except Exception:
        return None


def read_mot(path, is_gt=False):
    by=defaultdict(list)
    if not Path(path).exists(): return by
    for line in Path(path).read_text().splitlines():
        if not line.strip(): continue
        p=line.split(',')
        if len(p)<6: continue
        try:
            fr=int(float(p[0])); tid=int(float(p[1])); x=float(p[2]); y=float(p[3]); w=float(p[4]); h=float(p[5])
        except Exception:
            continue
        if w<=0 or h<=0: continue
        if is_gt:
            mark=int(float(p[6])) if len(p)>6 and p[6] else 1
            cls=int(float(p[7])) if len(p)>7 and p[7] else 1
            if mark<=0 or cls!=1: continue
        by[fr].append({'frame':fr,'id':tid,'box':np.array([x,y,x+w,y+h], dtype=np.float32)})
    return by


def iou(a,b):
    if a is None or b is None: return 0.0
    lt=np.maximum(a[:2], b[:2]); rb=np.minimum(a[2:], b[2:])
    wh=np.clip(rb-lt, 0, None); inter=float(wh[0]*wh[1])
    aa=max(1e-6, float((a[2]-a[0])*(a[3]-a[1]))); bb=max(1e-6, float((b[2]-b[0])*(b[3]-b[1])))
    return inter/max(1e-6, aa+bb-inter)


def box_arr(rows):
    return np.stack([r['box'] for r in rows], axis=0) if rows else np.zeros((0,4), dtype=np.float32)


def pair_iou(a,b):
    if len(a)==0 or len(b)==0: return np.zeros((len(a),len(b)), dtype=np.float32)
    lt=np.maximum(a[:,None,:2], b[None,:,:2]); rb=np.minimum(a[:,None,2:], b[None,:,2:])
    wh=np.clip(rb-lt,0,None); inter=wh[...,0]*wh[...,1]
    aa=np.clip((a[:,2]-a[:,0])*(a[:,3]-a[:,1]), 1e-6, None)
    bb=np.clip((b[:,2]-b[:,0])*(b[:,3]-b[:,1]), 1e-6, None)
    return inter/np.clip(aa[:,None]+bb[None,:]-inter,1e-6,None)


def build_track_identity_timeline(track_file, gt_root, seq, iou_thr=0.5, needed_by_frame=None):
    tr=read_mot(track_file, is_gt=False); gt=read_mot(Path(gt_root)/seq/'gt'/'gt.txt', is_gt=True)
    needed_by_frame = needed_by_frame or {}
    frames=sorted(set(tr)|set(gt)|set(needed_by_frame))
    # before_gt: last-known identity before current-frame update. For candidate tracks
    # that are lost and not output at current frame, we still snapshot their memory.
    before_gt={}; before_gap={}; current_gt={}; switch_keys=set(); idsw_keys=set()
    last_gt={}; last_track_for_gt={}
    for fr in frames:
        tracks=tr.get(fr, []); gts=gt.get(fr, [])
        for tid in needed_by_frame.get(fr, set()):
            prev = last_gt.get(int(tid), (-1, -1))
            before_gt[(fr, int(tid))] = prev[0]
            before_gap[(fr, int(tid))] = (fr - prev[1]) if prev[1] >= 0 else -1
        for t in tracks:
            tid = int(t['id'])
            if (fr, tid) not in before_gt:
                prev = last_gt.get(tid, (-1, -1))
                before_gt[(fr, tid)] = prev[0]
                before_gap[(fr, tid)] = (fr - prev[1]) if prev[1] >= 0 else -1
        if tracks and gts:
            I=pair_iou(box_arr(tracks), box_arr(gts))
            ri,ci=linear_sum_assignment(1-I)
            for r,c in zip(ri,ci):
                val=float(I[r,c])
                if val<iou_thr: continue
                tid=int(tracks[r]['id']); gid=int(gts[c]['id'])
                current_gt[(fr,tid)] = gid
                prev=last_gt.get(tid)
                if prev and prev[0]!=gid:
                    switch_keys.add((fr,tid))
                last_gt[tid]=(gid,fr)
                prev_track=last_track_for_gt.get(gid)
                if prev_track and prev_track[0]!=tid:
                    idsw_keys.add((fr,tid))
                last_track_for_gt[gid]=(tid,fr)
    return before_gt,before_gap,current_gt,switch_keys,idsw_keys,gt


def det_gt_from_box(det_box, gt_rows, thr=0.5):
    best=(-1,0.0)
    for g in gt_rows:
        val=iou(det_box, g['box'])
        if val>best[1]: best=(int(g['id']), val)
    return best if best[1]>=thr else (-1,best[1])


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--candidate-csv', required=True)
    ap.add_argument('--track-file', required=True)
    ap.add_argument('--gt-root', required=True)
    ap.add_argument('--seq', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--iou-thr', type=float, default=0.5)
    args=ap.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    groups=defaultdict(list)
    needed_by_frame=defaultdict(set)
    # group by frame+det_id and collect candidate track ids for memory snapshots
    with open(args.candidate_csv, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            fr=int(float(row['frame'])); det_id=int(float(row['det_id'])); tid=int(float(row['track_id']))
            groups[(fr,det_id)].append(row)
            needed_by_frame[fr].add(tid)

    before_gt,before_gap,current_gt,switch_keys,idsw_keys,gt_by_frame=build_track_identity_timeline(args.track_file,args.gt_root,args.seq,args.iou_thr, needed_by_frame=needed_by_frame)

    event_rows=[]
    summary=defaultdict(int)
    by_kind=defaultdict(lambda: defaultdict(int))
    bad_true_gaps=[]; idsw_true_gaps=[]; switch_true_gaps=[]
    for (fr,det_id), rows in groups.items():
        rows.sort(key=lambda r:int(float(r.get('candidate_rank',0))))
        chosen=[r for r in rows if int(float(r.get('is_chosen',0)))==1]
        chosen_row=chosen[0] if chosen else rows[0]
        chosen_tid=int(float(chosen_row['track_id']))
        det_box=parse_tlwh(chosen_row.get('det_tlwh',''))
        det_gt, det_iou=det_gt_from_box(det_box, gt_by_frame.get(fr, []), args.iou_thr)
        chosen_before=before_gt.get((fr,chosen_tid), -1)
        chosen_current=current_gt.get((fr,chosen_tid), -1)
        correct_chosen_before=int(det_gt>=0 and chosen_before==det_gt)
        correct_chosen_current=int(det_gt>=0 and chosen_current==det_gt)
        contains_before=0; contains_current=0; rank_before=-1; rank_current=-1
        best_true_cost=999.0; true_last_seen_gap=-1
        candidate_tids=[]
        for r in rows:
            tid=int(float(r['track_id'])); rank=int(float(r['candidate_rank'])); cost=float(r['candidate_cost'])
            candidate_tids.append(tid)
            if det_gt>=0 and before_gt.get((fr,tid), -1)==det_gt:
                contains_before=1
                if rank_before<0 or rank<rank_before:
                    rank_before=rank
                    true_last_seen_gap=before_gap.get((fr,tid), -1)
                best_true_cost=min(best_true_cost,cost)
            if det_gt>=0 and current_gt.get((fr,tid), -1)==det_gt:
                contains_current=1
                if rank_current<0 or rank<rank_current: rank_current=rank
        is_switch=int((fr,chosen_tid) in switch_keys)
        is_idsw=int((fr,chosen_tid) in idsw_keys)
        is_bad_commit=int(det_gt>=0 and chosen_before>=0 and chosen_before!=det_gt)
        low_margin_005=int(float(chosen_row.get('col_margin',999))<=0.05 or float(chosen_row.get('row_margin',999))<=0.05)
        rec={
            'seq':args.seq,'frame':fr,'det_id':det_id,'det_gt':det_gt,'det_iou':det_iou,
            'chosen_tid':chosen_tid,'chosen_before_gt':chosen_before,'chosen_current_gt':chosen_current,
            'correct_chosen_before':correct_chosen_before,'correct_chosen_current':correct_chosen_current,
            'contains_true_before_topk':contains_before,'rank_true_before':rank_before,'true_last_seen_gap':true_last_seen_gap,
            'contains_true_current_topk':contains_current,'rank_true_current':rank_current,
            'is_track_switch':is_switch,'is_gt_idsw':is_idsw,'is_bad_commit_before':is_bad_commit,
            'low_margin_005':low_margin_005,
            'chosen_cost':float(chosen_row.get('candidate_cost',999)),
            'best_true_cost':best_true_cost if contains_before else '',
            'row_margin':float(chosen_row.get('row_margin',999)),'col_margin':float(chosen_row.get('col_margin',999)),
            'num_candidates':len(rows),'candidate_tids':'|'.join(map(str,candidate_tids[:10]))
        }
        event_rows.append(rec)
        # summary on matched detections only
        if det_gt>=0:
            summary['events_matched_det_gt']+=1
            summary['chosen_before_correct']+=correct_chosen_before
            summary['contains_true_before_topk']+=contains_before
            summary['chosen_current_correct']+=correct_chosen_current
            summary['contains_true_current_topk']+=contains_current
            if is_bad_commit:
                summary['bad_commit_before']+=1
                summary['bad_commit_contains_true_before_topk']+=contains_before
                summary['bad_commit_low_margin_005']+=low_margin_005
                if contains_before: bad_true_gaps.append(true_last_seen_gap)
            if is_switch:
                summary['track_switch_events']+=1
                summary['track_switch_contains_true_before_topk']+=contains_before
                summary['track_switch_low_margin_005']+=low_margin_005
                if contains_before: switch_true_gaps.append(true_last_seen_gap)
            if is_idsw:
                summary['gt_idsw_events']+=1
                summary['gt_idsw_contains_true_before_topk']+=contains_before
                summary['gt_idsw_low_margin_005']+=low_margin_005
                if contains_before: idsw_true_gaps.append(true_last_seen_gap)

    def rate(a,b): return float(a)/float(b) if b else 0.0
    def pct(vals, q):
        if not vals: return None
        return float(np.percentile(np.asarray(vals, dtype=float), q))

    derived={
        'chosen_before_accuracy':rate(summary['chosen_before_correct'], summary['events_matched_det_gt']),
        'contains_true_before_topk_rate':rate(summary['contains_true_before_topk'], summary['events_matched_det_gt']),
        'bad_commit_contains_true_before_topk_rate':rate(summary['bad_commit_contains_true_before_topk'], summary['bad_commit_before']),
        'bad_commit_low_margin_005_rate':rate(summary['bad_commit_low_margin_005'], summary['bad_commit_before']),
        'track_switch_contains_true_before_topk_rate':rate(summary['track_switch_contains_true_before_topk'], summary['track_switch_events']),
        'gt_idsw_contains_true_before_topk_rate':rate(summary['gt_idsw_contains_true_before_topk'], summary['gt_idsw_events']),
        'gt_idsw_low_margin_005_rate':rate(summary['gt_idsw_low_margin_005'], summary['gt_idsw_events']),
        'bad_commit_true_last_seen_gap_p50':pct(bad_true_gaps, 50),
        'bad_commit_true_last_seen_gap_p90':pct(bad_true_gaps, 90),
        'track_switch_true_last_seen_gap_p50':pct(switch_true_gaps, 50),
        'track_switch_true_last_seen_gap_p90':pct(switch_true_gaps, 90),
        'gt_idsw_true_last_seen_gap_p50':pct(idsw_true_gaps, 50),
        'gt_idsw_true_last_seen_gap_p90':pct(idsw_true_gaps, 90),
    }
    # write summary/sample
    with open(out/'candidate_contains_true_events.csv','w',newline='',encoding='utf-8') as f:
        fields=list(event_rows[0].keys()) if event_rows else []
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(event_rows)
    # samples bad/switch/idsw
    for name, pred in [
        ('bad_commit_before_sample', lambda r: int(r['is_bad_commit_before'])==1),
        ('track_switch_sample', lambda r: int(r['is_track_switch'])==1),
        ('gt_idsw_sample', lambda r: int(r['is_gt_idsw'])==1),
    ]:
        rr=[r for r in event_rows if pred(r)][:5000]
        with open(out/f'{name}.csv','w',newline='',encoding='utf-8') as f:
            fields=list(event_rows[0].keys()) if event_rows else []
            w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rr)
    payload={'seq':args.seq,'summary_counts':dict(summary),'derived_rates':derived,'notes':{
        'before_gt':'candidate track identity before current commit; this is the relevant identity for escrow/defer decisions.',
        'current_gt':'identity after current frame matching; less useful for pre-commit decisions.'
    }}
    (out/'candidate_contains_true_summary.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
    md=['# A37 candidate contains true audit','',f"seq: {args.seq}",'','## Counts','```json',json.dumps(dict(summary),indent=2,sort_keys=True),'```','','## Rates']
    for k,v in derived.items(): md.append(f'- {k}: {v:.4f}')
    (out/'candidate_contains_true_summary.md').write_text('\n'.join(md)+'\n')
    print(json.dumps(payload,indent=2,sort_keys=True))

if __name__=='__main__': main()

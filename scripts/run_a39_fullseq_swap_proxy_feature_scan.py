#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "external/BoT-SORT-main"))
from fast_reid.fast_reid_interfece import FastReIDInterface  # noqa: E402


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


def read_csv(path: Path):
    if not path.exists():
        return []
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows):
    fields=[]; seen=set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k); fields.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=fields or ['transaction_id'], extrasaction='ignore')
        w.writeheader(); w.writerows(rows)


def l2norm(v):
    return v / max(float(np.linalg.norm(v)), 1e-12)


def read_track(path: Path):
    rows=[]; by_frame_tid=defaultdict(list); by_tid=defaultdict(list)
    with path.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            p=line.rstrip('\n').split(',')
            fr=ai(p[0],-1); tid=ai(p[1],-1); x=af(p[2]); y=af(p[3]); w=af(p[4]); h=af(p[5]); score=af(p[6],1.0) if len(p)>6 else 1.0
            if fr < 0 or tid < 0 or w <= 0 or h <= 0:
                continue
            r={'idx':len(rows),'parts':p,'frame':fr,'track_id':tid,'x':x,'y':y,'w':w,'h':h,'score':score,'box':np.array([x,y,x+w,y+h], dtype=np.float32)}
            rows.append(r); by_frame_tid[(fr,tid)].append(r); by_tid[tid].append(r)
    for tid in by_tid:
        by_tid[tid]=sorted(by_tid[tid], key=lambda r:r['frame'])
    return rows, by_frame_tid, by_tid


def choose_rows(rows, n):
    return sorted(sorted(rows, key=lambda r:(-r.get('score',1.0), r['frame']))[:n], key=lambda r:r['frame'])


def img_path(img_dir: Path, frame: int):
    return img_dir / f"{frame:06d}.jpg"


def extract_features(items, img_dir: Path, encoder):
    by_frame=defaultdict(list)
    for k,r in items:
        by_frame[r['frame']].append((k,r))
    feats={}
    for fr, vals in sorted(by_frame.items()):
        img=cv2.imread(str(img_path(img_dir, fr)))
        if img is None:
            continue
        dets=np.stack([r['box'] for _,r in vals]).astype(np.float32)
        out=encoder.inference(img, dets)
        for (k,_), feat in zip(vals, out):
            feats[k]=l2norm(feat.astype(np.float32))
    return feats


def mean_proto(vs):
    vs=[v for v in vs if v is not None]
    if not vs:
        return None
    return l2norm(np.mean(np.stack(vs, axis=0), axis=0))


def parse_segments(s):
    out=[]
    for part in str(s or '').split('|'):
        if not part:
            continue
        if '-' in part:
            a,b=part.split('-',1); out.append((ai(a),ai(b)))
        else:
            fr=ai(part); out.append((fr,fr))
    return out


def frames_from_segments(segs):
    out=[]
    for a,b in segs:
        out.extend(range(a,b+1))
    return sorted(set(out))


def segments(frames):
    if not frames:
        return []
    frames=sorted(frames); out=[]; s=e=frames[0]
    for fr in frames[1:]:
        if fr == e+1:
            e=fr
        else:
            out.append((s,e)); s=e=fr
    out.append((s,e)); return out


def seg_str(segs):
    return '|'.join(f'{a}-{b}' if a != b else str(a) for a,b in segs)


def rows_for(by_tid, tid, lo, hi):
    return [r for r in by_tid.get(tid, []) if lo <= r['frame'] <= hi]


def paired_rows_for(by_frame_tid, ta, tb, frames):
    pairs=[]
    for fr in frames:
        ar=by_frame_tid.get((fr,ta),[]); br=by_frame_tid.get((fr,tb),[])
        if not ar or not br:
            continue
        a=sorted(ar, key=lambda r:-r['score'])[0]
        b=sorted(br, key=lambda r:-r['score'])[0]
        pairs.append((fr,a,b))
    return pairs


def viterbi(features, penalty=0.1):
    n=len(features)
    if n == 0:
        return []
    dp=np.zeros((n,2), dtype=np.float64); prev=np.zeros((n,2), dtype=np.int64)
    dp[0,0]=features[0]['no_score']; dp[0,1]=features[0]['swap_score']
    for i in range(1,n):
        for s in [0,1]:
            emit=features[i]['swap_score'] if s else features[i]['no_score']
            vals=[dp[i-1,ps] - (penalty if ps != s else 0.0) + emit for ps in [0,1]]
            prev[i,s]=int(np.argmax(vals)); dp[i,s]=max(vals)
    states=[0]*n; states[-1]=int(np.argmax(dp[-1]))
    for i in range(n-1,0,-1):
        states[i-1]=int(prev[i,states[i]])
    return states


def stat(vals, fn, d=0.0):
    return float(fn(vals)) if vals else d


def region_stats(name, rows, feats, proto_correct, proto_alt):
    sims_c=[]; sims_a=[]; margins=[]
    for r in rows:
        v=feats.get(f'row_{r["idx"]}')
        if v is None or proto_correct is None or proto_alt is None:
            continue
        sc=float(np.dot(v, proto_correct)); sa=float(np.dot(v, proto_alt)); m=sc-sa
        sims_c.append(sc); sims_a.append(sa); margins.append(m)
    return {
        f'{name}_rows':len(rows),
        f'{name}_feat_rows':len(margins),
        f'{name}_sim_correct_mean':stat(sims_c,np.mean),
        f'{name}_sim_alt_mean':stat(sims_a,np.mean),
        f'{name}_margin_mean':stat(margins,np.mean),
        f'{name}_margin_min':stat(margins,np.min),
        f'{name}_margin_p10':stat(margins,lambda x:np.percentile(x,10)),
    }


def region_proto(rows, feats):
    return mean_proto([feats.get(f'row_{r["idx"]}') for r in rows])


def sim_proto(a,b):
    if a is None or b is None:
        return 0.0
    return float(np.dot(a,b))


def accepted_key_set(path: Path):
    keys=set()
    for r in read_csv(path):
        keys.add((r.get('frame'), r.get('old_id'), r.get('new_id')))
    return keys


def score_swap(rec, m_pre=0.15, m_swap=0.08, m_post=0.05, m_boundary=0.60, min_rows=5, min_frames=20):
    reasons=[]
    if ai(rec.get('changed_rows')) <= 0: reasons.append('no_swap_rows')
    if ai(rec.get('duplicate_covered_by_other')): reasons.append('duplicate_covered')
    if ai(rec.get('pred_segments_count')) != 1: reasons.append('fragmented_or_multi_segment')
    if ai(rec.get('pred_swap_frames')) < min_frames: reasons.append('too_few_swap_frames')
    if not ai(rec.get('proto_a_available')) or not ai(rec.get('proto_b_available')): reasons.append('missing_proto')
    for region in ['a_pre_A','b_pre_B','a_swap_B','b_swap_A','a_post_B','b_post_A']:
        if ai(rec.get(f'{region}_feat_rows')) < min_rows:
            reasons.append(f'{region}_too_few_feat_rows')
    for region in ['a_pre_A','b_pre_B']:
        if af(rec.get(f'{region}_margin_mean')) < m_pre:
            reasons.append(f'{region}_margin_low')
    for region in ['a_swap_B','b_swap_A']:
        if af(rec.get(f'{region}_margin_mean')) < m_swap:
            reasons.append(f'{region}_margin_low')
    for region in ['a_post_B','b_post_A']:
        if af(rec.get(f'{region}_margin_mean')) < m_post:
            reasons.append(f'{region}_margin_low')
    if af(rec.get('boundary_min_sim')) < m_boundary:
        reasons.append('boundary_sim_low')
    return int(not reasons), '|'.join(reasons) if reasons else 'persistent_handoff_proxy_v1_nogt'


def apply_swap_track(rows, state_by_frame, ta, tb):
    parts=[r['parts'][:] for r in rows]
    audit=[]
    for r in rows:
        fr=r['frame']; tid=r['track_id']
        if state_by_frame.get(fr,0) != 1 or tid not in {ta,tb}:
            continue
        new_id = tb if tid == ta else ta
        parts[r['idx']][1]=str(new_id)
        audit.append({'frame':fr,'idx':r['idx'],'old_track_id':tid,'new_track_id':new_id})
    return parts,audit


def write_track(path: Path, parts):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(','.join(p) for p in parts)+'\n', encoding='utf-8')


def feature_one(job, rows, by_frame_tid, by_tid, img_dir, encoder, proto_window, proto_crops, accepted_keys, out_case_dir):
    tid=job['transaction_id']; ta=ai(job['track_a']); tb=ai(job['track_b'])
    ov_frames=frames_from_segments(parse_segments(job.get('overlap_segments','')))
    if not ov_frames:
        ov_frames=list(range(ai(job['overlap_start']), ai(job['overlap_end'])+1))
    pairs=paired_rows_for(by_frame_tid, ta, tb, ov_frames)
    if not pairs:
        return {'transaction_id':tid,'status':'no_paired_rows','nogt_accept':0,'nogt_reason':'no_paired_rows'}
    fs=min(fr for fr,_,_ in pairs); fe=max(fr for fr,_,_ in pairs)
    pre_a=choose_rows(rows_for(by_tid, ta, fs-proto_window, fs-1), proto_crops)
    pre_b=choose_rows(rows_for(by_tid, tb, fs-proto_window, fs-1), proto_crops)
    # Extract features for prototypes and paired interval first.
    items=[]
    for r in pre_a+pre_b:
        items.append((f'row_{r["idx"]}', r))
    for _,a,b in pairs:
        items.append((f'row_{a["idx"]}', a)); items.append((f'row_{b["idx"]}', b))
    # Include post rows for lifecycle proxy.
    post_a=rows_for(by_tid, ta, fe+1, fe+proto_window)
    post_b=rows_for(by_tid, tb, fe+1, fe+proto_window)
    for r in post_a+post_b:
        items.append((f'row_{r["idx"]}', r))
    # Deduplicate.
    dedup={k:r for k,r in items}
    feats=extract_features(list(dedup.items()), img_dir, encoder)
    proto_A=region_proto(pre_a, feats); proto_B=region_proto(pre_b, feats)
    if proto_A is None or proto_B is None:
        return {'transaction_id':tid,'status':'missing_proto','nogt_accept':0,'nogt_reason':'missing_proto'}
    feat_rows=[]
    for fr,a,b in pairs:
        fa=feats.get(f'row_{a["idx"]}'); fb=feats.get(f'row_{b["idx"]}')
        if fa is None or fb is None:
            continue
        no=float(np.dot(fa, proto_A)+np.dot(fb, proto_B))
        sw=float(np.dot(fa, proto_B)+np.dot(fb, proto_A))
        feat_rows.append({'frame':fr,'no_score':no,'swap_score':sw,'swap_margin':sw-no})
    states=viterbi(feat_rows, penalty=0.1)
    swap_frames=[r['frame'] for r,s in zip(feat_rows,states) if s==1]
    swap_segs=segments(swap_frames)
    state_by_frame={r['frame']:s for r,s in zip(feat_rows,states)}
    # Lifecycle regions under persistent handoff.
    a_pre=rows_for(by_tid, ta, fs-proto_window, fs-1)
    b_pre=rows_for(by_tid, tb, fs-proto_window, fs-1)
    a_swap=[a for fr,a,b in pairs if state_by_frame.get(fr,0)==1]
    b_swap=[b for fr,a,b in pairs if state_by_frame.get(fr,0)==1]
    a_post=post_a; b_post=post_b
    rec={'transaction_id':tid,'track_a':ta,'track_b':tb,'candidate_family':'swap','mode':'swap_persistent_handoff','overlap_start':fs,'overlap_end':fe,'overlap_frames':len(pairs),'pred_segments':seg_str(swap_segs),'pred_segments_count':len(swap_segs),'pred_swap_frames':len(swap_frames),'changed_rows':len(swap_frames)*2,'proto_a_available':1,'proto_b_available':1}
    for name, rr, corr, alt in [
        ('a_pre_A',a_pre,proto_A,proto_B),('b_pre_B',b_pre,proto_B,proto_A),('a_swap_B',a_swap,proto_B,proto_A),('b_swap_A',b_swap,proto_A,proto_B),('a_post_B',a_post,proto_B,proto_A),('b_post_A',b_post,proto_A,proto_B)]:
        rec.update(region_stats(name, rr, feats, corr, alt))
    p_a_pre=region_proto(a_pre, feats); p_b_pre=region_proto(b_pre, feats)
    p_a_swap=region_proto(a_swap, feats); p_b_swap=region_proto(b_swap, feats)
    p_a_post=region_proto(a_post, feats); p_b_post=region_proto(b_post, feats)
    rec['slot_A_pre_to_swap_sim']=sim_proto(p_a_pre,p_b_swap)
    rec['slot_A_swap_to_post_sim']=sim_proto(p_b_swap,p_b_post)
    rec['slot_B_pre_to_swap_sim']=sim_proto(p_b_pre,p_a_swap)
    rec['slot_B_swap_to_post_sim']=sim_proto(p_a_swap,p_a_post)
    rec['boundary_min_sim']=min(rec['slot_A_pre_to_swap_sim'],rec['slot_A_swap_to_post_sim'],rec['slot_B_pre_to_swap_sim'],rec['slot_B_swap_to_post_sim'])
    parts,audit=apply_swap_track(rows, state_by_frame, ta, tb)
    # duplicate if every changed row maps exactly to an already accepted current-best change.
    if audit:
        covered=sum(1 for x in audit if (str(x['frame']), str(x['old_track_id']), str(x['new_track_id'])) in accepted_keys)
        rec['duplicate_covered_by_other']=int(covered == len(audit))
        rec['covered_rows_by_other']=covered
    else:
        rec['duplicate_covered_by_other']=0; rec['covered_rows_by_other']=0
    acc,reason=score_swap(rec)
    rec['nogt_accept']=acc; rec['nogt_reason']=reason
    if 'duplicate_covered' in reason:
        rec['stage']='duplicate_covered'
    else:
        rec['stage']='accepted_new' if acc else 'scorer_rejected'
    out_case_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_case_dir/'swap_row_audit_nogt.csv', audit)
    write_track(out_case_dir/'track_results'/'MOT20-02.txt', parts)
    (out_case_dir/'feature_summary.json').write_text(json.dumps(rec, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    write_csv(out_case_dir/'state_features.csv', feat_rows)
    return rec


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--stage-report', default='outputs/spot_runtime_gate_20260628/A39_identity_suspension_tunnel/A39_06d_full_seq_no_gt_candidate_scan_seq02/transaction_stage_report_full_nogt_cached.csv')
    ap.add_argument('--track-file', default='outputs/spot_runtime_gate_20260628/A37_identity_escrow_association/A37_00b_candidate_logger_seq02/track_results/MOT20-02.txt')
    ap.add_argument('--img-dir', default='datasets/MOT20/train/MOT20-02/img1')
    ap.add_argument('--fast-reid-config', default='external/BoT-SORT-main/fast_reid/configs/MOT20/sbs_S50.yml')
    ap.add_argument('--fast-reid-weights', default='external/BoT-SORT-main/pretrained/mot20_sbs_S50.pth')
    ap.add_argument('--accepted-change-audit', default='outputs/spot_runtime_gate_20260628/A39_identity_suspension_tunnel/A39_05c_swap_state_segmentation_for_reciprocal_swap/combined_12_202_106reid_215reid/combined_change_audit.csv')
    ap.add_argument('--out-dir', default='outputs/spot_runtime_gate_20260628/A39_identity_suspension_tunnel/A39_06d_full_seq_no_gt_candidate_scan_seq02/A39_06d2b_swap_prefilter_heavy_feature_scan')
    ap.add_argument('--mode', choices=['manifest','smoke','full'], default='smoke')
    ap.add_argument('--top-k', type=int, default=30)
    ap.add_argument('--max-contiguous-overlap', type=int, default=0, help='If >0, keep only jobs with max_contiguous_overlap <= this value')
    ap.add_argument('--max-track-count', type=int, default=0, help='If >0, keep only jobs with track_count_in_tunnel <= this value')
    ap.add_argument('--exclude-completed-cases', action='store_true', help='Skip jobs whose case feature_summary.json already exists in out_dir')
    ap.add_argument('--proto-window', type=int, default=50)
    ap.add_argument('--proto-crops', type=int, default=24)
    ap.add_argument('--device', default='cuda')
    args=ap.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    missing=[r for r in read_csv(Path(args.stage_report)) if r.get('stage')=='prefilter_pass_feature_missing']
    def priority(r):
        ov=ai(r.get('max_contiguous_overlap') or r.get('overlap_frames'))
        tc=ai(r.get('track_count_in_tunnel'),99)
        # lower track count and longer overlap first; no GT.
        return (ov, -tc, -ai(r.get('tunnel_id')), -ai(r.get('track_a')), -ai(r.get('track_b')))
    missing=sorted(missing, key=priority, reverse=True)
    if args.max_contiguous_overlap > 0:
        missing=[r for r in missing if ai(r.get('max_contiguous_overlap') or r.get('overlap_frames')) <= args.max_contiguous_overlap]
    if args.max_track_count > 0:
        missing=[r for r in missing if ai(r.get('track_count_in_tunnel'), 999) <= args.max_track_count]
    if args.exclude_completed_cases:
        missing=[r for r in missing if not (out/'cases'/r['transaction_id']/'feature_summary.json').exists()]
    for i,r in enumerate(missing):
        r['job_rank']=i+1
        r['batch_id']=(i//50)+1
        r['job_status']='pending'
    write_csv(out/'swap_feature_job_manifest.csv', missing)
    if args.mode == 'manifest':
        print(json.dumps({'job_count':len(missing),'out_dir':str(out)}, indent=2))
        return
    critical_ids={'106_169_150','215_508_469','12_7_10','12_10_7','12_11_13','12_13_21'}
    jobs=[]; seen=set()
    for r in missing:
        if r['transaction_id'] in critical_ids:
            jobs.append(r); seen.add(r['transaction_id'])
    for r in missing:
        if len(jobs) >= args.top_k:
            break
        if r['transaction_id'] not in seen:
            jobs.append(r); seen.add(r['transaction_id'])
    if args.mode == 'full':
        jobs=missing
    write_csv(out/'swap_feature_jobs_selected.csv', jobs)
    rows, by_frame_tid, by_tid=read_track(Path(args.track_file))
    encoder=FastReIDInterface(args.fast_reid_config, args.fast_reid_weights, args.device, batch_size=32)
    acc_keys=accepted_key_set(Path(args.accepted_change_audit))
    feats=[]; failed=[]
    for idx,job in enumerate(jobs, start=1):
        tid=job['transaction_id']
        try:
            rec=feature_one(job, rows, by_frame_tid, by_tid, Path(args.img_dir), encoder, args.proto_window, args.proto_crops, acc_keys, out/'cases'/tid)
            rec.update({k:job.get(k,'') for k in ['job_rank','batch_id','tunnel_id','track_count_in_tunnel','overlap_segments','max_contiguous_overlap']})
            feats.append(rec)
            print(f'[{idx}/{len(jobs)}] {tid} {rec.get("stage")} {rec.get("nogt_reason")}')
        except Exception as e:
            failed.append({'transaction_id':tid,'stage':'feature_failed','error':repr(e)})
            print(f'[{idx}/{len(jobs)}] {tid} FAILED {e}')
    write_csv(out/'swap_proxy_features_scanned.csv', feats)
    write_csv(out/'swap_proxy_scan_failed.csv', failed)
    accepted=[r for r in feats if ai(r.get('nogt_accept'))]
    duplicate=[r for r in feats if r.get('stage')=='duplicate_covered']
    rejected=[r for r in feats if not ai(r.get('nogt_accept')) and r.get('stage')!='duplicate_covered']
    write_csv(out/'swap_proxy_accepted_new.csv', accepted)
    write_csv(out/'swap_proxy_rejected_scanned.csv', rejected)
    write_csv(out/'swap_proxy_duplicate_covered_scanned.csv', duplicate)
    summary={'mode':args.mode,'job_count_total':len(missing),'job_count_selected':len(jobs),'feature_rows':len(feats),'failed':len(failed),'accepted_new':[r['transaction_id'] for r in accepted],'duplicate_covered':[r['transaction_id'] for r in duplicate],'stage_counts':dict(Counter(r.get('stage') for r in feats))}
    (out/'swap_proxy_scan_summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    print(json.dumps(summary, indent=2, sort_keys=True))

if __name__ == '__main__':
    main()

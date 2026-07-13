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


def safe_div(a, b):
    return float(a) / float(b) if b else 0.0


def read_csv(path: Path):
    if not path.exists(): return []
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
        w=csv.DictWriter(f, fieldnames=fields or ['case'], extrasaction='ignore')
        w.writeheader(); w.writerows(rows)


def read_track(path: Path):
    rows=[]; by_frame=defaultdict(list); by_frame_tid=defaultdict(list); by_tid=defaultdict(list)
    with path.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            p=line.strip().split(',')
            fr=ai(p[0],-1); tid=ai(p[1],-1); x=af(p[2]); y=af(p[3]); w=af(p[4]); h=af(p[5]); score=af(p[6],1.0) if len(p)>6 else 1.0
            if fr<0 or tid<0 or w<=0 or h<=0: continue
            r={'idx':len(rows),'frame':fr,'track_id':tid,'x':x,'y':y,'w':w,'h':h,'score':score,'box':np.array([x,y,x+w,y+h],dtype=np.float32)}
            rows.append(r); by_frame[fr].append(r); by_frame_tid[(fr,tid)].append(r); by_tid[tid].append(r)
    return rows,by_frame,by_frame_tid,by_tid


def read_gt(path: Path):
    by=defaultdict(list)
    with path.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            p=line.strip().split(',')
            fr=ai(p[0],-1); gid=ai(p[1],-1); x=af(p[2]); y=af(p[3]); w=af(p[4]); h=af(p[5])
            mark=ai(p[6],1) if len(p)>6 else 1; cls=ai(p[7],1) if len(p)>7 else 1
            if fr<0 or gid<0 or w<=0 or h<=0 or mark<=0 or cls!=1: continue
            by[fr].append({'frame':fr,'gt_id':gid,'box':np.array([x,y,x+w,y+h],dtype=np.float32)})
    return by


def pair_iou(a,b):
    if not a or not b: return np.zeros((len(a),len(b)),dtype=np.float32)
    A=np.stack([x['box'] for x in a]); B=np.stack([x['box'] for x in b])
    lt=np.maximum(A[:,None,:2],B[None,:,:2]); rb=np.minimum(A[:,None,2:],B[None,:,2:])
    wh=np.clip(rb-lt,0,None); inter=wh[...,0]*wh[...,1]
    aa=np.clip((A[:,2]-A[:,0])*(A[:,3]-A[:,1]),1e-6,None); bb=np.clip((B[:,2]-B[:,0])*(B[:,3]-B[:,1]),1e-6,None)
    return inter/np.clip(aa[:,None]+bb[None,:]-inter,1e-6,None)


def match_rows(rows_by, gt_by, thr):
    row_gt={}; row_iou={}
    for fr in sorted(set(rows_by)|set(gt_by)):
        rr=rows_by.get(fr,[]); gg=gt_by.get(fr,[])
        if not rr or not gg: continue
        I=pair_iou(rr,gg); ri,ci=linear_sum_assignment(1-I)
        for r,c in zip(ri,ci):
            val=float(I[r,c])
            if val>=thr:
                row_gt[rr[r]['idx']]=int(gg[c]['gt_id']); row_iou[rr[r]['idx']]=val
    return row_gt,row_iou


def parse_segments(s: str):
    out=[]
    if not s: return out
    for part in str(s).split('|'):
        if not part: continue
        if '-' in part:
            a,b=part.split('-',1); out.append((ai(a),ai(b)))
        else:
            fr=ai(part); out.append((fr,fr))
    return out


def frames_from_segments(segs):
    frames=[]
    for a,b in segs:
        frames.extend(range(a,b+1))
    return sorted(set(frames))


def gt_counts(rows, row_gt):
    c=Counter()
    for r in rows:
        gid=row_gt.get(r['idx'],-1)
        if gid>=0: c[gid]+=1
    matched=sum(c.values())
    major_gt,major_n=(c.most_common(1)[0] if c else (-1,0))
    return c,matched,major_gt,major_n


def support_for(rows, expected_gt, row_gt):
    c,matched,major_gt,major_n=gt_counts(rows,row_gt)
    return {
        'rows': len(rows),
        'matched_rows': matched,
        'expected_gt': expected_gt,
        'support_rows': c.get(expected_gt,0),
        'support_ratio': safe_div(c.get(expected_gt,0), matched),
        'major_gt': major_gt,
        'major_purity': safe_div(major_n, matched),
    }


def rows_for(by_frame_tid, tid, frames):
    out=[]
    for fr in frames:
        out.extend(by_frame_tid.get((fr,tid),[]))
    return out


def discover_state_summaries(roots, methods):
    out=[]
    allowed=set(methods) if methods else None
    for root in roots:
        root=Path(root)
        if not root.exists(): continue
        for p in root.glob('*/*/state_summary.json'):
            method=p.parent.name
            if allowed and method not in allowed: continue
            try:
                data=json.loads(p.read_text())
            except Exception:
                continue
            data['state_summary_path']=str(p)
            data['state_root']=str(root)
            out.append(data)
    return out


def read_metrics_for_summary(s):
    d=Path(s['state_summary_path']).parent
    for p in d.glob('eval_mot20_02/eval/*/pedestrian_summary.txt'):
        lines=p.read_text().strip().splitlines()
        if len(lines)>=2:
            return dict(zip(lines[0].split(), lines[1].split()))
    return {}


def audit_case(s, by_frame_tid, row_gt, accepted_keys, window):
    case=s['case_name']; method=s['method']; ta=ai(s['track_a']); tb=ai(s['track_b']); gta=ai(s['gt_a']); gtb=ai(s['gt_b'])
    pred_segs=parse_segments(s.get('pred_segments',''))
    pred_frames=frames_from_segments(pred_segs)
    pred_start=min(pred_frames) if pred_frames else ai(s.get('frame_start'))
    pred_end=max(pred_frames) if pred_frames else ai(s.get('frame_end'))
    pre_frames=list(range(pred_start-window, pred_start))
    post_frames=list(range(pred_end+1, pred_end+window+1))
    swap_frames=pred_frames
    # Slot A: track_a pre, track_b swap, track_a post should all be gt_a.
    a_pre=support_for(rows_for(by_frame_tid, ta, pre_frames), gta, row_gt)
    a_swap=support_for(rows_for(by_frame_tid, tb, swap_frames), gta, row_gt)
    a_post=support_for(rows_for(by_frame_tid, ta, post_frames), gta, row_gt)
    b_pre=support_for(rows_for(by_frame_tid, tb, pre_frames), gtb, row_gt)
    b_swap=support_for(rows_for(by_frame_tid, ta, swap_frames), gtb, row_gt)
    b_post=support_for(rows_for(by_frame_tid, tb, post_frames), gtb, row_gt)
    # Coverage against accepted current-best transactions.
    audit_path=Path(s['state_summary_path']).parent/'swap_row_audit.csv'
    audit_rows=read_csv(audit_path)
    swap_keys={(r['frame'],r['old_track_id'],r['new_track_id']) for r in audit_rows}
    covered=len(swap_keys & accepted_keys)
    duplicate_covered=int(len(swap_keys)>0 and covered==len(swap_keys))
    metrics=read_metrics_for_summary(s)
    pred_segments_count=len(pred_segs)
    # Labels for report.
    label='negative'
    if case in {'106_150_169','215_469_508'} and method.startswith('reid_'):
        label='positive'
    if case=='214_508_469':
        label='duplicate_covered'
    # GT lifecycle gate v1: strict, explanatory.
    reasons=[]
    if ai(s.get('changed_rows')) <= 0: reasons.append('no_swap_rows')
    if ai(s.get('wrong_after_swap_rows')) > 1: reasons.append('wrong_rows')
    if ai(s.get('same_frame_duplicate_count_after_swap')) > 0: reasons.append('duplicates')
    if duplicate_covered: reasons.append('duplicate_covered')
    if pred_segments_count != 1: reasons.append('fragmented_or_multi_segment')
    if ai(s.get('pred_swap_frames')) < 20: reasons.append('too_few_swap_frames')
    # Require both slots recover after local transaction. For swap mode, no post recovery is risky.
    supports={
        'slot_A_pre':a_pre['support_ratio'], 'slot_A_swap':a_swap['support_ratio'], 'slot_A_post':a_post['support_ratio'],
        'slot_B_pre':b_pre['support_ratio'], 'slot_B_swap':b_swap['support_ratio'], 'slot_B_post':b_post['support_ratio'],
    }
    rows_counts={
        'slot_A_pre':a_pre['matched_rows'], 'slot_A_swap':a_swap['matched_rows'], 'slot_A_post':a_post['matched_rows'],
        'slot_B_pre':b_pre['matched_rows'], 'slot_B_swap':b_swap['matched_rows'], 'slot_B_post':b_post['matched_rows'],
    }
    for k,v in supports.items():
        if rows_counts[k] < 5:
            reasons.append(f'{k}_too_few_rows')
        elif v < 0.80:
            reasons.append(f'{k}_low_support')
    pass_gate=int(not reasons)
    rec={
        'case':case,'method':method,'label':label,'track_a':ta,'track_b':tb,'gt_a':gta,'gt_b':gtb,
        'pred_segments':s.get('pred_segments',''),'pred_segments_count':pred_segments_count,'pred_start':pred_start,'pred_end':pred_end,
        'changed_rows':s.get('changed_rows'),'wrong_after_swap_rows':s.get('wrong_after_swap_rows'),'duplicate_count':s.get('same_frame_duplicate_count_after_swap'),
        'swap_precision':s.get('swap_precision'),'swap_recall':s.get('swap_recall'),
        'HOTA':metrics.get('HOTA',''),'IDF1':metrics.get('IDF1',''),'IDSW':metrics.get('IDSW',''),'MOTA':metrics.get('MOTA',''),'Frag':metrics.get('Frag',''),
        'swap_rows_total':len(audit_rows),'swap_rows_covered_by_current_best':covered,'duplicate_covered_by_current_best':duplicate_covered,
        'lifecycle_gate_gt_v1_pass':pass_gate,'lifecycle_failure_reason':'|'.join(reasons) if reasons else 'pass',
    }
    for prefix,st in [('track_a_pre',a_pre),('track_b_swap_for_A',a_swap),('track_a_post',a_post),('track_b_pre',b_pre),('track_a_swap_for_B',b_swap),('track_b_post',b_post)]:
        for k,v in st.items(): rec[f'{prefix}_{k}']=v
    return rec


def report_rule(rows, rule_field):
    # duplicate label is neither TP nor FP; keep separate.
    tp=fp=tn=fn=dup=0; accepted=[]; rejected=[]
    for r in rows:
        if r['label']=='duplicate_covered':
            dup+=1
            continue
        pred=ai(r.get(rule_field))
        lab=1 if r['label']=='positive' else 0
        if pred and lab: tp+=1; accepted.append(r['case']+':'+r['method'])
        elif pred and not lab: fp+=1; accepted.append(r['case']+':'+r['method'])
        elif not pred and not lab: tn+=1; rejected.append(r['case']+':'+r['method'])
        else: fn+=1; rejected.append(r['case']+':'+r['method'])
    return {'rule_name':rule_field,'tp':tp,'fp':fp,'tn':tn,'fn':fn,'duplicate_covered':dup,'precision':safe_div(tp,tp+fp),'recall':safe_div(tp,tp+fn),'accepted':'|'.join(accepted),'rejected':'|'.join(rejected)}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--track-file', default='outputs/spot_runtime_gate_20260628/A37_identity_escrow_association/A37_00b_candidate_logger_seq02/track_results/MOT20-02.txt')
    ap.add_argument('--gt-file', default='datasets/MOT20/train/MOT20-02/gt/gt.txt')
    ap.add_argument('--state-root', action='append', required=True)
    ap.add_argument('--accepted-change-audit', default='outputs/spot_runtime_gate_20260628/A39_identity_suspension_tunnel/A39_05c_swap_state_segmentation_for_reciprocal_swap/combined_12_202_106reid_215reid/combined_change_audit.csv')
    ap.add_argument('--out-dir', default='outputs/spot_runtime_gate_20260628/A39_identity_suspension_tunnel/A39_05e_global_lifecycle_gate_for_swap_transactions')
    ap.add_argument('--methods', default='reid_viterbi_penalty_0.1,reid_raw_margin_gt_0,gt_state_upper_bound')
    ap.add_argument('--prepost-window', type=int, default=50)
    ap.add_argument('--iou-thr', type=float, default=0.5)
    args=ap.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rows, rows_by, by_frame_tid, by_tid = read_track(Path(args.track_file))
    gt_by=read_gt(Path(args.gt_file)); row_gt,row_iou=match_rows(rows_by, gt_by, args.iou_thr)
    accepted_rows=read_csv(Path(args.accepted_change_audit))
    accepted_keys={(r['frame'],r['old_id'],r['new_id']) for r in accepted_rows}
    methods=[x.strip() for x in args.methods.split(',') if x.strip()]
    summaries=discover_state_summaries(args.state_root, methods)
    audit=[audit_case(s, by_frame_tid, row_gt, accepted_keys, args.prepost_window) for s in summaries]
    audit=sorted(audit, key=lambda r:(r['case'], r['method']))
    write_csv(out/'swap_lifecycle_audit.csv', audit)
    report=[report_rule([r for r in audit if r['method']=='reid_viterbi_penalty_0.1'], 'lifecycle_gate_gt_v1_pass'), report_rule(audit, 'lifecycle_gate_gt_v1_pass')]
    write_csv(out/'lifecycle_gate_report.csv', report)
    payload={'audit_rows':len(audit),'report':report}
    (out/'summary.json').write_text(json.dumps(payload, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    md=['# A39_05e Global Lifecycle Gate Audit','','## Summary','','```json',json.dumps(payload, indent=2, sort_keys=True),'```','','## ReID viterbi 0.1 lifecycle audit','','| case | label | gate | reason | seg | HOTA | IDF1 | IDSW | Apre | Aswap | Apost | Bpre | Bswap | Bpost |','|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in audit:
        if r['method']!='reid_viterbi_penalty_0.1': continue
        md.append(f"| {r['case']} | {r['label']} | {r['lifecycle_gate_gt_v1_pass']} | {r['lifecycle_failure_reason']} | {r['pred_segments']} | {r['HOTA']} | {r['IDF1']} | {r['IDSW']} | {af(r.get('track_a_pre_support_ratio')):.2f} | {af(r.get('track_b_swap_for_A_support_ratio')):.2f} | {af(r.get('track_a_post_support_ratio')):.2f} | {af(r.get('track_b_pre_support_ratio')):.2f} | {af(r.get('track_a_swap_for_B_support_ratio')):.2f} | {af(r.get('track_b_post_support_ratio')):.2f} |")
    (out/'decision_draft.md').write_text('\n'.join(md)+'\n', encoding='utf-8')
    print('\n'.join(md))

if __name__=='__main__':
    main()

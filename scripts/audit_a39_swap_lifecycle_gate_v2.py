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
    try: return int(float(v))
    except Exception: return d


def af(v, d=0.0):
    try: return float(v)
    except Exception: return d


def safe_div(a,b): return float(a)/float(b) if b else 0.0


def read_csv(path: Path):
    if not path.exists(): return []
    with path.open(newline='', encoding='utf-8') as f: return list(csv.DictReader(f))


def write_csv(path: Path, rows):
    fields=[]; seen=set()
    for r in rows:
        for k in r:
            if k not in seen: seen.add(k); fields.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=fields or ['case'], extrasaction='ignore')
        w.writeheader(); w.writerows(rows)


def read_track(path: Path):
    rows=[]; by_frame=defaultdict(list); by_frame_tid=defaultdict(list)
    with path.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            p=line.strip().split(',')
            fr=ai(p[0],-1); tid=ai(p[1],-1); x=af(p[2]); y=af(p[3]); w=af(p[4]); h=af(p[5])
            if fr<0 or tid<0 or w<=0 or h<=0: continue
            r={'idx':len(rows),'frame':fr,'track_id':tid,'box':np.array([x,y,x+w,y+h],dtype=np.float32)}
            rows.append(r); by_frame[fr].append(r); by_frame_tid[(fr,tid)].append(r)
    return rows,by_frame,by_frame_tid


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
    row_gt={}
    for fr in sorted(set(rows_by)|set(gt_by)):
        rr=rows_by.get(fr,[]); gg=gt_by.get(fr,[])
        if not rr or not gg: continue
        I=pair_iou(rr,gg); ri,ci=linear_sum_assignment(1-I)
        for r,c in zip(ri,ci):
            if float(I[r,c])>=thr: row_gt[rr[r]['idx']]=int(gg[c]['gt_id'])
    return row_gt


def parse_segments(s):
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
    f=[]
    for a,b in segs: f.extend(range(a,b+1))
    return sorted(set(f))


def rows_for(by_frame_tid, tid, frames):
    out=[]
    for fr in frames: out.extend(by_frame_tid.get((fr,tid),[]))
    return out


def support(rows, expected_gt, row_gt):
    c=Counter(row_gt.get(r['idx'],-1) for r in rows if row_gt.get(r['idx'],-1)>=0)
    matched=sum(c.values()); major,major_n=(c.most_common(1)[0] if c else (-1,0))
    return {'rows':len(rows),'matched':matched,'support':c.get(expected_gt,0),'ratio':safe_div(c.get(expected_gt,0),matched),'major':major,'purity':safe_div(major_n,matched)}


def discover(roots, methods):
    allowed=set(methods)
    out=[]
    for root in roots:
        root=Path(root)
        for p in root.glob('*/*/state_summary.json'):
            if p.parent.name not in allowed: continue
            try: s=json.loads(p.read_text())
            except Exception: continue
            s['state_summary_path']=str(p); out.append(s)
    return out


def metrics(s):
    d=Path(s['state_summary_path']).parent
    for p in d.glob('eval_mot20_02/eval/*/pedestrian_summary.txt'):
        lines=p.read_text().strip().splitlines()
        if len(lines)>=2: return dict(zip(lines[0].split(),lines[1].split()))
    return {}


def accepted_source_map(path: Path):
    mp=defaultdict(list)
    for r in read_csv(path):
        mp[(r['frame'],r['old_id'],r['new_id'])].append(r.get('source_anchor',''))
    return mp


def is_duplicate_covered(case, audit_rows, acc_map):
    if not audit_rows: return 0,0
    case_prefix=case.split('_')[0]
    covered=0
    for r in audit_rows:
        key=(r['frame'],r['old_track_id'],r['new_track_id'])
        sources=acc_map.get(key,[])
        # Covered by a different accepted transaction, not by itself.
        if any(not str(src).startswith(case_prefix) for src in sources):
            covered+=1
    return int(covered==len(audit_rows)), covered


def label_for(case, method):
    if case in {'106_150_169','215_469_508'} and method.startswith('reid_'): return 'positive'
    if case=='214_508_469': return 'duplicate_covered'
    return 'negative'


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--track-file', default='outputs/spot_runtime_gate_20260628/A37_identity_escrow_association/A37_00b_candidate_logger_seq02/track_results/MOT20-02.txt')
    ap.add_argument('--gt-file', default='datasets/MOT20/train/MOT20-02/gt/gt.txt')
    ap.add_argument('--state-root', action='append', required=True)
    ap.add_argument('--accepted-change-audit', default='outputs/spot_runtime_gate_20260628/A39_identity_suspension_tunnel/A39_05c_swap_state_segmentation_for_reciprocal_swap/combined_12_202_106reid_215reid/combined_change_audit.csv')
    ap.add_argument('--out-dir', default='outputs/spot_runtime_gate_20260628/A39_identity_suspension_tunnel/A39_05e_global_lifecycle_gate_for_swap_transactions')
    ap.add_argument('--methods', default='reid_viterbi_penalty_0.1,reid_raw_margin_gt_0,gt_state_upper_bound')
    ap.add_argument('--window', type=int, default=50)
    ap.add_argument('--iou-thr', type=float, default=0.5)
    args=ap.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    _,rows_by,by_frame_tid=read_track(Path(args.track_file)); row_gt=match_rows(rows_by,read_gt(Path(args.gt_file)),args.iou_thr)
    acc=accepted_source_map(Path(args.accepted_change_audit))
    methods=[x.strip() for x in args.methods.split(',') if x.strip()]
    audits=[]
    for s in discover(args.state_root, methods):
        case=s['case_name']; method=s['method']; ta=ai(s['track_a']); tb=ai(s['track_b']); gta=ai(s['gt_a']); gtb=ai(s['gt_b'])
        segs=parse_segments(s.get('pred_segments','')); swap_frames=frames_from_segments(segs)
        ps=min(swap_frames) if swap_frames else ai(s.get('frame_start')); pe=max(swap_frames) if swap_frames else ai(s.get('frame_end'))
        pre=list(range(ps-args.window,ps)); post=list(range(pe+1,pe+args.window+1))
        # Pre should be original, swap should be swapped, post is allowed/expected to be persistent handoff.
        a_pre=support(rows_for(by_frame_tid,ta,pre),gta,row_gt)
        b_pre=support(rows_for(by_frame_tid,tb,pre),gtb,row_gt)
        a_swap_for_B=support(rows_for(by_frame_tid,ta,swap_frames),gtb,row_gt)
        b_swap_for_A=support(rows_for(by_frame_tid,tb,swap_frames),gta,row_gt)
        a_post_for_B=support(rows_for(by_frame_tid,ta,post),gtb,row_gt)
        b_post_for_A=support(rows_for(by_frame_tid,tb,post),gta,row_gt)
        audit_rows=read_csv(Path(s['state_summary_path']).parent/'swap_row_audit.csv')
        dup,covered=is_duplicate_covered(case,audit_rows,acc)
        m=metrics(s)
        reasons=[]
        if ai(s.get('changed_rows'))<=0: reasons.append('no_swap_rows')
        if ai(s.get('wrong_after_swap_rows'))>1: reasons.append('wrong_rows')
        if ai(s.get('same_frame_duplicate_count_after_swap'))>0: reasons.append('same_frame_duplicates')
        if dup: reasons.append('duplicate_covered')
        if len(segs)!=1: reasons.append('fragmented_or_multi_segment')
        if ai(s.get('pred_swap_frames'))<20: reasons.append('too_few_swap_frames')
        checks={
            'a_pre':a_pre,'b_pre':b_pre,'a_swap_for_B':a_swap_for_B,'b_swap_for_A':b_swap_for_A,'a_post_for_B':a_post_for_B,'b_post_for_A':b_post_for_A,
        }
        for k,st in checks.items():
            if st['matched']<5: reasons.append(f'{k}_too_few_rows')
            elif st['ratio']<0.80: reasons.append(f'{k}_low_support')
        gate=int(not reasons)
        rec={'case':case,'method':method,'label':label_for(case,method),'track_a':ta,'track_b':tb,'gt_a':gta,'gt_b':gtb,'pred_segments':s.get('pred_segments',''),'pred_segments_count':len(segs),'changed_rows':s.get('changed_rows'),'wrong_after_swap_rows':s.get('wrong_after_swap_rows'),'swap_precision':s.get('swap_precision'),'swap_recall':s.get('swap_recall'),'HOTA':m.get('HOTA',''),'IDF1':m.get('IDF1',''),'IDSW':m.get('IDSW',''),'MOTA':m.get('MOTA',''),'Frag':m.get('Frag',''),'duplicate_covered_by_other':dup,'covered_rows_by_other':covered,'persistent_handoff_gate_gt_v2_pass':gate,'lifecycle_failure_reason':'|'.join(reasons) if reasons else 'pass'}
        for name,st in checks.items():
            for k,v in st.items(): rec[f'{name}_{k}']=v
        audits.append(rec)
    audits=sorted(audits,key=lambda r:(r['case'],r['method']))
    write_csv(out/'swap_lifecycle_audit_v2.csv',audits)
    def report(rows):
        tp=fp=tn=fn=dup=0; accs=[]; rejs=[]
        for r in rows:
            if r['label']=='duplicate_covered': dup+=1; continue
            pred=ai(r['persistent_handoff_gate_gt_v2_pass']); lab=int(r['label']=='positive')
            if pred and lab: tp+=1; accs.append(r['case']+':'+r['method'])
            elif pred and not lab: fp+=1; accs.append(r['case']+':'+r['method'])
            elif not pred and not lab: tn+=1; rejs.append(r['case']+':'+r['method'])
            else: fn+=1; rejs.append(r['case']+':'+r['method'])
        return {'rule_name':'persistent_handoff_gate_gt_v2','tp':tp,'fp':fp,'tn':tn,'fn':fn,'duplicate_covered':dup,'precision':safe_div(tp,tp+fp),'recall':safe_div(tp,tp+fn),'accepted':'|'.join(accs),'rejected':'|'.join(rejs)}
    reports=[report([r for r in audits if r['method']=='reid_viterbi_penalty_0.1']), report(audits)]
    write_csv(out/'lifecycle_gate_report_v2.csv',reports)
    payload={'audit_rows':len(audits),'report':reports}
    (out/'summary_v2.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
    md=['# A39_05e Persistent Handoff Lifecycle Gate v2','','```json',json.dumps(payload,indent=2,sort_keys=True),'```','','| case | label | gate | reason | seg | HOTA | IDF1 | IDSW | Apre | Bpre | AswapB | BswapA | ApostB | BpostA |','|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in audits:
        if r['method']!='reid_viterbi_penalty_0.1': continue
        md.append(f"| {r['case']} | {r['label']} | {r['persistent_handoff_gate_gt_v2_pass']} | {r['lifecycle_failure_reason']} | {r['pred_segments']} | {r['HOTA']} | {r['IDF1']} | {r['IDSW']} | {af(r['a_pre_ratio']):.2f} | {af(r['b_pre_ratio']):.2f} | {af(r['a_swap_for_B_ratio']):.2f} | {af(r['b_swap_for_A_ratio']):.2f} | {af(r['a_post_for_B_ratio']):.2f} | {af(r['b_post_for_A_ratio']):.2f} |")
    (out/'decision_draft_v2.md').write_text('\n'.join(md)+'\n')
    print('\n'.join(md))

if __name__=='__main__': main()

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
from scipy.optimize import linear_sum_assignment

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


def safe_div(a, b):
    return float(a) / float(b) if b else 0.0


def read_track(path: Path):
    rows=[]; by_frame=defaultdict(list); by_frame_tid=defaultdict(list); by_tid=defaultdict(list)
    with path.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            p=line.rstrip('\n').split(',')
            fr=ai(p[0],-1); tid=ai(p[1],-1); x=af(p[2]); y=af(p[3]); w=af(p[4]); h=af(p[5]); score=af(p[6],1.0) if len(p)>6 else 1.0
            if fr<0 or tid<0 or w<=0 or h<=0:
                continue
            r={'idx':len(rows),'parts':p,'frame':fr,'track_id':tid,'x':x,'y':y,'w':w,'h':h,'score':score,'box':np.array([x,y,x+w,y+h],dtype=np.float32)}
            r['cx']=x+w/2.0; r['bottom_y']=y+h; r['height']=h
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


def img_path(img_dir: Path, frame: int):
    return img_dir / f"{frame:06d}.jpg"


def choose_rows(rows, n):
    return sorted(sorted(rows, key=lambda r:(-r.get('score',1.0), r['frame']))[:n], key=lambda r:r['frame'])


def extract_row_features(row_items, img_dir: Path, encoder):
    # row_items: list[(key,row)]
    by_frame=defaultdict(list)
    for k,r in row_items:
        by_frame[r['frame']].append((k,r))
    feats={}
    for fr,items in sorted(by_frame.items()):
        img=cv2.imread(str(img_path(img_dir,fr)))
        if img is None:
            continue
        dets=np.stack([r['box'] for _,r in items]).astype(np.float32)
        out=encoder.inference(img,dets)
        for (k,r),feat in zip(items,out):
            v=feat.astype(np.float32)
            v=v/max(float(np.linalg.norm(v)),1e-12)
            feats[k]=v
    return feats


def mean_proto(feats):
    if not feats: return None
    v=np.mean(np.stack(feats,axis=0),axis=0)
    return v/max(float(np.linalg.norm(v)),1e-12)


def parse_case(s: str):
    # name:tunnel:track_a:track_b:start:end:gt_a:gt_b
    p=s.split(':')
    if len(p)!=8:
        raise ValueError('case format name:tunnel:track_a:track_b:start:end:gt_a:gt_b')
    return {'case_name':p[0],'tunnel_id':ai(p[1]),'track_a':ai(p[2]),'track_b':ai(p[3]),'frame_start':ai(p[4]),'frame_end':ai(p[5]),'gt_a':ai(p[6]),'gt_b':ai(p[7])}


def viterbi(features, penalty):
    # features rows sorted by frame, contains no_score, swap_score. higher score better.
    n=len(features)
    if n==0: return []
    dp=np.zeros((n,2),dtype=np.float64); prev=np.zeros((n,2),dtype=np.int64)
    dp[0,0]=features[0]['no_score']; dp[0,1]=features[0]['swap_score']
    for i in range(1,n):
        for s in [0,1]:
            emit=features[i]['swap_score'] if s else features[i]['no_score']
            vals=[dp[i-1,ps] - (penalty if ps!=s else 0.0) + emit for ps in [0,1]]
            prev[i,s]=int(np.argmax(vals)); dp[i,s]=max(vals)
    states=[0]*n; states[-1]=int(np.argmax(dp[-1]))
    for i in range(n-1,0,-1): states[i-1]=int(prev[i,states[i]])
    return states


def segments(frames):
    if not frames: return []
    frames=sorted(frames); out=[]; s=frames[0]; e=frames[0]
    for fr in frames[1:]:
        if fr==e+1: e=fr
        else: out.append((s,e)); s=e=fr
    out.append((s,e)); return out


def apply_state(rows, case, state_by_frame, row_gt):
    new_parts=[r['parts'][:] for r in rows]
    audit=[]; changed=set()
    for r in rows:
        fr=r['frame']; tid=r['track_id']
        if state_by_frame.get(fr,0)!=1: continue
        if tid not in {case['track_a'],case['track_b']}: continue
        if tid==case['track_a']:
            new_id=case['track_b']; expected=case['gt_b']
        else:
            new_id=case['track_a']; expected=case['gt_a']
        new_parts[r['idx']][1]=str(new_id); changed.add(r['idx'])
        gid=row_gt.get(r['idx'],-1)
        audit.append({'case':case['case_name'],'frame':fr,'idx':r['idx'],'old_track_id':tid,'new_track_id':new_id,'row_gt':gid,'expected_gt_after_swap':expected,'is_correct_after_swap':int(gid==expected) if gid>=0 else 0})
    # duplicate after swap
    c=Counter((ai(p[0]),ai(p[1])) for p in new_parts)
    dup=[(fr,tid,n) for (fr,tid),n in c.items() if n>1]
    correct=sum(int(x['is_correct_after_swap']) for x in audit)
    wrong=sum(1 for x in audit if ai(x['row_gt'],-1)>=0 and int(x['is_correct_after_swap'])==0)
    unknown=sum(1 for x in audit if ai(x['row_gt'],-1)<0)
    return new_parts,audit,{'changed_rows':len(audit),'correct_after_swap_rows':correct,'wrong_after_swap_rows':wrong,'unknown_rows':unknown,'same_frame_duplicate_count_after_swap':len(dup),'duplicate_examples':dup[:20]}


def write_track(path: Path, parts):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text('\n'.join(','.join(p) for p in parts)+'\n',encoding='utf-8')


def write_csv(path: Path, rows):
    fields=[]; seen=set()
    for r in rows:
        for k in r:
            if k not in seen: seen.add(k); fields.append(k)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields or ['case']); w.writeheader(); w.writerows(rows)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--track-file', required=True)
    ap.add_argument('--gt-file', required=True)
    ap.add_argument('--img-dir', required=True)
    ap.add_argument('--fast-reid-config', required=True)
    ap.add_argument('--fast-reid-weights', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--case', action='append', required=True, help='name:tunnel:track_a:track_b:start:end:gt_a:gt_b')
    ap.add_argument('--proto-window', type=int, default=50)
    ap.add_argument('--proto-crops', type=int, default=24)
    ap.add_argument('--penalties', default='0.0,0.05,0.10,0.20')
    ap.add_argument('--margin-thr', type=float, default=0.0)
    ap.add_argument('--iou-thr', type=float, default=0.5)
    ap.add_argument('--device', default='cuda')
    args=ap.parse_args()

    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    rows,rows_by,by_frame_tid,by_tid=read_track(Path(args.track_file))
    gt_by=read_gt(Path(args.gt_file)); row_gt,row_iou=match_rows(rows_by,gt_by,args.iou_thr)
    cases=[parse_case(x) for x in args.case]
    encoder=FastReIDInterface(args.fast_reid_config,args.fast_reid_weights,args.device,batch_size=32)
    all_summary=[]
    all_features=[]
    for case in cases:
        cname=case['case_name']; fs=case['frame_start']; fe=case['frame_end']; ta=case['track_a']; tb=case['track_b']
        proto_a_rows=[r for r in by_tid[ta] if fs-args.proto_window <= r['frame'] < fs]
        proto_b_rows=[r for r in by_tid[tb] if fs-args.proto_window <= r['frame'] < fs]
        proto_a_rows=choose_rows(proto_a_rows,args.proto_crops); proto_b_rows=choose_rows(proto_b_rows,args.proto_crops)
        # interval paired rows
        frames=[]; interval_items=[]; row_a_by_frame={}; row_b_by_frame={}
        for fr in range(fs,fe+1):
            ar=by_frame_tid.get((fr,ta),[]); br=by_frame_tid.get((fr,tb),[])
            if not ar or not br: continue
            a=sorted(ar,key=lambda r:-r['score'])[0]; b=sorted(br,key=lambda r:-r['score'])[0]
            frames.append(fr); row_a_by_frame[fr]=a; row_b_by_frame[fr]=b
            interval_items.append((f'{cname}_a_{fr}',a)); interval_items.append((f'{cname}_b_{fr}',b))
        proto_items=[(f'{cname}_proto_a_{i}',r) for i,r in enumerate(proto_a_rows)] + [(f'{cname}_proto_b_{i}',r) for i,r in enumerate(proto_b_rows)]
        feats=extract_row_features(proto_items+interval_items,Path(args.img_dir),encoder)
        proto_a=mean_proto([feats[k] for k,_ in proto_items if '_proto_a_' in k and k in feats])
        proto_b=mean_proto([feats[k] for k,_ in proto_items if '_proto_b_' in k and k in feats])
        if proto_a is None or proto_b is None:
            raise RuntimeError(f'missing prototype for {cname}')
        feat_rows=[]
        for fr in frames:
            a=row_a_by_frame[fr]; b=row_b_by_frame[fr]
            fa=feats.get(f'{cname}_a_{fr}'); fb=feats.get(f'{cname}_b_{fr}')
            if fa is None or fb is None: continue
            sim_a_A=float(np.dot(fa,proto_a)); sim_a_B=float(np.dot(fa,proto_b)); sim_b_A=float(np.dot(fb,proto_a)); sim_b_B=float(np.dot(fb,proto_b))
            no_score=sim_a_A+sim_b_B; swap_score=sim_a_B+sim_b_A
            a_gt=row_gt.get(a['idx'],-1); b_gt=row_gt.get(b['idx'],-1)
            gt_state = 1 if (a_gt==case['gt_b'] and b_gt==case['gt_a']) else 0
            gt_clean = int((a_gt==case['gt_b'] and b_gt==case['gt_a']) or (a_gt==case['gt_a'] and b_gt==case['gt_b']))
            rec={'case':cname,'frame':fr,'track_a':ta,'track_b':tb,'row_a_idx':a['idx'],'row_b_idx':b['idx'],'row_a_gt':a_gt,'row_b_gt':b_gt,'gt_state_swap':gt_state,'gt_clean_state':gt_clean,
                 'sim_a_A':sim_a_A,'sim_a_B':sim_a_B,'sim_b_A':sim_b_A,'sim_b_B':sim_b_B,'no_score':no_score,'swap_score':swap_score,'swap_margin':swap_score-no_score,
                 'a_score':a['score'],'b_score':b['score'],'a_height':a['height'],'b_height':b['height'],'height_ratio_a_b':a['height']/max(b['height'],1e-6),'center_dist_norm':abs(a['cx']-b['cx'])/max(max(a['height'],b['height']),1e-6),'bottom_dist_norm':abs(a['bottom_y']-b['bottom_y'])/max(max(a['height'],b['height']),1e-6)}
            feat_rows.append(rec); all_features.append(rec)
        # state methods
        methods=[]
        gt_state_by={r['frame']:int(r['gt_state_swap']) for r in feat_rows}
        methods.append(('gt_state_upper_bound',gt_state_by))
        raw={r['frame']:int(r['swap_margin']>args.margin_thr) for r in feat_rows}
        methods.append((f'reid_raw_margin_gt_{args.margin_thr:g}',raw))
        for pen in [af(x) for x in args.penalties.split(',') if x.strip()!='']:
            states=viterbi(feat_rows,pen)
            methods.append((f'reid_viterbi_penalty_{pen:g}',{r['frame']:s for r,s in zip(feat_rows,states)}))
        for mname,state_by in methods:
            mdir=out/cname/mname
            parts,audit,diag=apply_state(rows,case,state_by,row_gt)
            write_track(mdir/'track_results'/'MOT20-02.txt',parts)
            write_csv(mdir/'swap_row_audit.csv',audit)
            swap_frames=[fr for fr,s in state_by.items() if s==1]
            gt_swap_frames=[r['frame'] for r in feat_rows if int(r['gt_state_swap'])==1]
            pred=set(swap_frames); gtset=set(gt_swap_frames); allf=set(r['frame'] for r in feat_rows)
            tp=len(pred&gtset); fp=len(pred-gtset); fn=len(gtset-pred); tn=len(allf-pred-gtset)
            summary={**case,'method':mname,**diag,'pred_swap_frames':len(pred),'gt_swap_frames':len(gtset),'swap_precision':safe_div(tp,tp+fp),'swap_recall':safe_div(tp,tp+fn),'frame_accuracy':safe_div(tp+tn,len(allf)),'tp_frames':tp,'fp_frames':fp,'fn_frames':fn,'tn_frames':tn,'pred_segments':'|'.join(f'{s}-{e}' for s,e in segments(swap_frames)),'gt_segments':'|'.join(f'{s}-{e}' for s,e in segments(gt_swap_frames))}
            (mdir/'state_summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n',encoding='utf-8')
            all_summary.append(summary)
        write_csv(out/cname/'swap_state_features.csv',feat_rows)
    write_csv(out/'swap_state_features_all.csv',all_features)
    write_csv(out/'swap_state_segment_summary.csv',all_summary)
    (out/'summary.json').write_text(json.dumps({'cases':[c['case_name'] for c in cases],'method_count':len(all_summary),'summary_csv':str(out/'swap_state_segment_summary.csv')},indent=2)+'\n')
    print(json.dumps({'method_count':len(all_summary),'out_dir':str(out)},indent=2))

if __name__=='__main__':
    main()

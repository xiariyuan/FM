#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, math, shutil, subprocess, sys
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment


def fnum(x, default=0.0):
    try: return float(x)
    except Exception: return default

def inum(x, default=0):
    try: return int(float(x))
    except Exception: return default

def read_mot_rows(path: Path, is_gt=False) -> Dict[int, List[dict]]:
    by=defaultdict(list)
    with path.open('r',encoding='utf-8',errors='ignore') as fh:
        for line in fh:
            line=line.strip()
            if not line: continue
            p=line.split(',')
            if len(p)<6: continue
            fr=inum(p[0]); tid=inum(p[1]); x=fnum(p[2]); y=fnum(p[3]); w=fnum(p[4]); h=fnum(p[5])
            if w<=0 or h<=0: continue
            score=fnum(p[6],1.0) if len(p)>6 else 1.0
            if is_gt:
                mark=inum(p[6],1) if len(p)>6 else 1
                cls=inum(p[7],1) if len(p)>7 else 1
                if mark==0 or cls!=1: continue
                vis=fnum(p[8],1.0) if len(p)>8 else 1.0
                score=1.0
            else:
                vis=1.0
            by[fr].append({'frame':fr,'id':tid,'x':x,'y':y,'w':w,'h':h,'score':score,'vis':vis,'tail':p[7:] if len(p)>7 else ['-1','-1','-1']})
    return by

def flatten(by):
    return [r for fr in sorted(by) for r in by[fr]]

def boxes(rows):
    arr=np.zeros((len(rows),4),dtype=np.float32)
    for i,r in enumerate(rows): arr[i]=[r['x'],r['y'],r['x']+r['w'],r['y']+r['h']]
    return arr

def iou_mat(a,b):
    if not a or not b: return np.zeros((len(a),len(b)),dtype=np.float32)
    A=boxes(a); B=boxes(b)
    ax1,ay1,ax2,ay2=A[:,0:1],A[:,1:2],A[:,2:3],A[:,3:4]
    bx1,by1,bx2,by2=B[:,0][None,:],B[:,1][None,:],B[:,2][None,:],B[:,3][None,:]
    ix1=np.maximum(ax1,bx1); iy1=np.maximum(ay1,by1); ix2=np.minimum(ax2,bx2); iy2=np.minimum(ay2,by2)
    inter=np.maximum(0,ix2-ix1)*np.maximum(0,iy2-iy1)
    aa=np.maximum(0,ax2-ax1)*np.maximum(0,ay2-ay1); ba=np.maximum(0,bx2-bx1)*np.maximum(0,by2-by1)
    u=aa+ba-inter
    return np.where(u>0, inter/u, 0.0)

def match_gt(gt, det, thr=0.5):
    if not gt: return set(), set(), []
    if not det: return set(), set(range(len(gt))), []
    M=iou_mat(gt,det); rr,cc=linear_sum_assignment(-M)
    matched_gt=set(); matched_det=set(); pairs=[]
    for r,c in zip(rr,cc):
        v=float(M[r,c])
        if v>=thr:
            matched_gt.add(r); matched_det.add(c); pairs.append((r,c,v))
    return matched_gt, set(range(len(gt)))-matched_gt, pairs

def area(r): return max(0,r['w'])*max(0,r['h'])
def center(r): return (r['x']+r['w']/2, r['y']+r['h']/2)
def dist(a,b):
    ax,ay=center(a); bx,by=center(b); return ((ax-bx)**2+(ay-by)**2)**0.5
def area_bucket_from_area(a):
    if a < 32*32: return 'small'
    if a < 96*96: return 'medium'
    return 'large'
def area_bucket(r): return area_bucket_from_area(area(r))

def write_mot(path: Path, rows: List[dict]):
    rows=sorted(rows,key=lambda r:(r['frame'],r['id'],r['x'],r['y']))
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8') as f:
        for r in rows:
            tail=r.get('tail') or ['-1','-1','-1']
            while len(tail)<3: tail.append('-1')
            f.write(f"{int(r['frame'])},{int(r['id'])},{r['x']:.2f},{r['y']:.2f},{r['w']:.2f},{r['h']:.2f},{r.get('score',1.0):.2f},{tail[0]},{tail[1]},{tail[2]}\n")

def parse_summary(path: Path):
    lines=[x.strip() for x in path.read_text(errors='ignore').splitlines() if x.strip()]
    if len(lines)<2: return {}
    d=dict(zip(lines[0].split(),lines[1].split()))
    out={}
    for k,v in d.items():
        try: out[k]=float(v)
        except: out[k]=v
    return out

def build_candidates(base_by, sent_by, gt_by, base_gt_matched, max_pool_overlap=0.5):
    segments=[]; cand_by_track=defaultdict(list)
    frames=sorted(set(sent_by)|set(base_by)|set(gt_by))
    for fr in frames:
        b=base_by.get(fr,[]); s=sent_by.get(fr,[]); g=gt_by.get(fr,[])
        if not s: continue
        sb=iou_mat(s,b) if b else np.zeros((len(s),0),dtype=np.float32)
        sg=iou_mat(s,g) if g else np.zeros((len(s),0),dtype=np.float32)
        maxb=sb.max(axis=1) if sb.shape[1] else np.zeros(len(s))
        bestg=sg.argmax(axis=1) if sg.shape[1] else np.zeros(len(s),dtype=int)
        maxg=sg.max(axis=1) if sg.shape[1] else np.zeros(len(s))
        for i,r in enumerate(s):
            if float(maxb[i]) >= max_pool_overlap: continue
            rr=dict(r); rr['max_iou_base']=float(maxb[i]); rr['max_iou_gt']=float(maxg[i])
            if g and maxg[i]>=0.5:
                gt=g[int(bestg[i])]
                rr['gt_id']=gt['id']; rr['gt_key']=f"{fr}:{gt['id']}"; rr['is_gt_match']=1; rr['is_recovered_gt']=int((fr,gt['id']) not in base_gt_matched); rr['gt_area_bucket']=area_bucket(gt)
            else:
                rr['gt_id']=-1; rr['gt_key']=''; rr['is_gt_match']=0; rr['is_recovered_gt']=0; rr['gt_area_bucket']='none'
            cand_by_track[rr['id']].append(rr)
    sid=0
    for tid, rows in cand_by_track.items():
        rows=sorted(rows,key=lambda r:r['frame'])
        cur=[]; last=None
        for r in rows:
            if last is None or r['frame']<=last+1:
                cur.append(r)
            else:
                if cur:
                    segments.append(make_segment(sid,tid,cur)); sid+=1
                cur=[r]
            last=r['frame']
        if cur:
            segments.append(make_segment(sid,tid,cur)); sid+=1
    return segments

def make_segment(sid, tid, rows):
    scores=[r['score'] for r in rows]; areas=[area(r) for r in rows]
    rec=sum(r['is_recovered_gt'] for r in rows); gt=sum(r['is_gt_match'] for r in rows); fp=len(rows)-gt
    ab=Counter(area_bucket(r) for r in rows).most_common(1)[0][0]
    return {'seg_id':sid,'sent_id':tid,'rows':rows,'len':len(rows),'start':rows[0]['frame'],'end':rows[-1]['frame'],'span':rows[-1]['frame']-rows[0]['frame']+1,'mean_score':sum(scores)/len(scores),'min_score':min(scores),'max_score':max(scores),'mean_area':sum(areas)/len(areas),'area_bucket':ab,'mean_iou_base':sum(r['max_iou_base'] for r in rows)/len(rows),'max_iou_base':max(r['max_iou_base'] for r in rows),'gt_frames':gt,'recovered_frames':rec,'fp_like_frames':fp,'precision_like':gt/len(rows),'recovered_precision_like':rec/len(rows)}

def find_inherit_id(seg, base_by, window=90, max_step=80.0, max_area_ratio=4.0):
    rows=seg['rows']; first=rows[0]; last=rows[-1]
    best=None
    for anchor, direction in [(first,'before'),(last,'after')]:
        if direction=='before': frames=range(max(1,anchor['frame']-window), anchor['frame'])
        else: frames=range(anchor['frame']+1, anchor['frame']+window+1)
        for fr in frames:
            gap=abs(anchor['frame']-fr)
            if gap<=0: continue
            for b in base_by.get(fr,[]):
                ar=max(area(anchor),area(b))/max(1e-6,min(area(anchor),area(b)))
                if ar>max_area_ratio: continue
                step=dist(anchor,b)/gap
                if step>max_step: continue
                score=step + 8.0*abs(math.log(max(1e-6,anchor['h'])/max(1e-6,b['h']))) + 2.0*abs(math.log(max(1e-6,area(anchor))/max(1e-6,area(b))))
                if best is None or score<best[0]: best=(score,b['id'],direction,fr,step,ar)
    return best[1] if best else None

def eval_policy(base_by, segs, out_policy, policy, args):
    base_rows=flatten(base_by); max_id=max(r['id'] for r in base_rows)
    selected=[]
    for seg in segs:
        if seg['len'] < policy['min_len']: continue
        if seg['mean_score'] < policy['min_mean_score']: continue
        if seg['min_score'] < policy['min_min_score']: continue
        if seg['max_iou_base'] >= policy['max_iou_base']: continue
        if policy.get('area') and seg['area_bucket'] not in policy['area']: continue
        selected.append(seg)
    add_rows=[]; inherit_count=0; new_count=0
    for idx,seg in enumerate(selected):
        if policy['id_mode']=='inherit':
            iid=find_inherit_id(seg, base_by, window=policy.get('inherit_window',90), max_step=policy.get('inherit_max_step',80), max_area_ratio=policy.get('inherit_max_area_ratio',4.0))
            if iid is None:
                iid=max_id+100000+idx; new_count+=1
            else:
                inherit_count+=1
        else:
            iid=max_id+100000+idx; new_count+=1
        for r in seg['rows']:
            rr=dict(r); rr['id']=iid; rr['tail']=['-1','-1','-1']; add_rows.append(rr)
    result_dir=out_policy/'data'; result_dir.mkdir(parents=True,exist_ok=True)
    write_mot(result_dir/'MOT20-05.txt', base_rows+add_rows)
    # Eval with halfval helper; input already remapped, so no remap flag.
    eval_dir=out_policy/'eval'
    tracker_name='A45_03_'+policy['name']
    cmd=[sys.executable,'scripts/eval_botsort_halfval_trackeval.py','--dataset','MOT20','--data-root','/gemini/code/datasets','--results-dir',str(result_dir),'--tracker-name',tracker_name,'--work-dir',str(eval_dir)]
    p=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (out_policy/'eval_stdout.log').write_text(p.stdout,encoding='utf-8')
    summary_path=eval_dir/'eval'/tracker_name/'pedestrian_summary.txt'
    m=parse_summary(summary_path) if summary_path.exists() else {'returncode':p.returncode}
    m['returncode']=p.returncode; m['summary_path']=str(summary_path)
    return {'policy':policy['name'],'id_mode':policy['id_mode'],'selected_segments':len(selected),'added_rows':len(add_rows),'inherit_segments':inherit_count,'new_segments':new_count,'selected_recovered_frames':sum(s['recovered_frames'] for s in selected),'selected_fp_like_frames':sum(s['fp_like_frames'] for s in selected),'selected_gt_frames':sum(s['gt_frames'] for s in selected),'metrics':m,'result_file':str(result_dir/'MOT20-05.txt')}

def write_csv(path, rows):
    fields=[]; seen=set()
    for r in rows:
        for k in r:
            if k not in seen: seen.add(k); fields.append(k)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields or ['x'],extrasaction='ignore'); w.writeheader(); w.writerows(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--gt',required=True); ap.add_argument('--baseline',required=True); ap.add_argument('--sentinel',required=True); ap.add_argument('--out-dir',required=True)
    args=ap.parse_args(); out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    gt_by=read_mot_rows(Path(args.gt),is_gt=True); base_by=read_mot_rows(Path(args.baseline)); sent_by=read_mot_rows(Path(args.sentinel))
    # baseline GT matched set
    base_matched=set()
    for fr in sorted(set(gt_by)|set(base_by)):
        gt=gt_by.get(fr,[]); b=base_by.get(fr,[]); mg,_,pairs=match_gt(gt,b,0.5)
        for gi,di,v in pairs: base_matched.add((fr,gt[gi]['id']))
    segs=build_candidates(base_by,sent_by,gt_by,base_matched,max_pool_overlap=0.5)
    # segment csv without rows
    seg_rows=[]
    for s in segs:
        seg_rows.append({k:v for k,v in s.items() if k!='rows'})
    write_csv(out/'a45_only_segments.csv',seg_rows)
    # Mine rule candidates by feature stats, and evaluate only concise hand-picked+best rules.
    policies=[]
    base_grid=[]
    for min_len in [5,10,20,30,50]:
        for mean_s in [0.03,0.05,0.1,0.2,0.3,0.5]:
            for max_iob in [0.1,0.3,0.5]:
                for area_sel in [None, ['medium'], ['medium','large']]:
                    selected=[s for s in segs if s['len']>=min_len and s['mean_score']>=mean_s and s['max_iou_base']<max_iob and (not area_sel or s['area_bucket'] in area_sel)]
                    rows=sum(s['len'] for s in selected); rec=sum(s['recovered_frames'] for s in selected); fp=sum(s['fp_like_frames'] for s in selected); gt=sum(s['gt_frames'] for s in selected)
                    if rows==0: continue
                    base_grid.append({'min_len':min_len,'min_mean_score':mean_s,'max_iou_base':max_iob,'area':'+'.join(area_sel) if area_sel else 'all','segments':len(selected),'rows':rows,'recovered':rec,'gt':gt,'fp_like':fp,'rec_per_row':rec/rows,'gt_per_row':gt/rows,'rec_minus_fp':rec-fp})
    base_grid=sorted(base_grid,key=lambda r:(r['rec_minus_fp'],r['recovered'],r['gt_per_row']),reverse=True)
    write_csv(out/'rule_mining_grid.csv',base_grid)
    # Select top unique plus deliberate conservative policies.
    chosen=[]; seen=set()
    def add_policy(name, min_len, mean_s, max_iob, area, id_mode, min_min_score=0.0):
        key=(min_len,mean_s,max_iob,tuple(area) if area else (),id_mode,min_min_score)
        if key in seen: return
        seen.add(key); chosen.append({'name':name,'min_len':min_len,'mean_score_label':mean_s,'min_mean_score':mean_s,'min_min_score':min_min_score,'max_iou_base':max_iob,'area':area,'id_mode':id_mode,'inherit_window':90,'inherit_max_step':80,'inherit_max_area_ratio':4.0})
    # top mined rules
    for i,r in enumerate(base_grid[:6]):
        area=None if r['area']=='all' else r['area'].split('+')
        add_policy(f"mine{i}_L{r['min_len']}_S{str(r['min_mean_score']).replace('.','p')}_O{str(r['max_iou_base']).replace('.','p')}_{r['area']}_new", int(r['min_len']), float(r['min_mean_score']), float(r['max_iou_base']), area, 'new')
        add_policy(f"mine{i}_L{r['min_len']}_S{str(r['min_mean_score']).replace('.','p')}_O{str(r['max_iou_base']).replace('.','p')}_{r['area']}_inherit", int(r['min_len']), float(r['min_mean_score']), float(r['max_iou_base']), area, 'inherit')
    # Hand policies
    for mode in ['new','inherit']:
        add_policy(f'conservative_L20_S0p2_O0p3_medlarge_{mode}',20,0.2,0.3,['medium','large'],mode)
        add_policy(f'conservative_L30_S0p1_O0p3_medlarge_{mode}',30,0.1,0.3,['medium','large'],mode)
        add_policy(f'wide_L10_S0p05_O0p5_all_{mode}',10,0.05,0.5,None,mode)
    results=[]
    for pol in chosen[:18]:
        res=eval_policy(base_by,segs,out/pol['name'],pol,args); results.append(res)
        print(json.dumps({'policy':res['policy'],'added_rows':res['added_rows'],'HOTA':res['metrics'].get('HOTA'),'IDF1':res['metrics'].get('IDF1'),'MOTA':res['metrics'].get('MOTA'),'IDSW':res['metrics'].get('IDSW'),'Frag':res['metrics'].get('Frag')},sort_keys=True),flush=True)
    # Flatten metrics
    rows=[]
    for r in results:
        m=r['metrics']; row={k:v for k,v in r.items() if k!='metrics'}
        for k in ['HOTA','DetA','AssA','MOTA','IDF1','CLR_Re','CLR_Pr','CLR_TP','CLR_FN','CLR_FP','IDSW','Frag','Dets','IDs']:
            row[k]=m.get(k,'')
        row['summary_path']=m.get('summary_path','')
        rows.append(row)
    write_csv(out/'a45_03_policy_eval.csv',rows)
    # Decision vs known halfval baseline.
    baseline={'HOTA':79.514784,'IDF1':91.34011979469209,'MOTA':94.10051964821884,'AssA':77.24652,'CLR_FN':13269,'CLR_FP':5223,'IDSW':331,'Frag':667}
    def val(row,k): return float(row.get(k) or 0)
    best=max(rows,key=lambda r:val(r,'HOTA')) if rows else {}
    decision='A45_03_NO_MERGE_BEATS_STRONG_BASELINE__STOP_A45_MERGE' if not best or val(best,'HOTA')<=baseline['HOTA'] else 'A45_03_FOUND_MERGE_BEATING_BASELINE__NEXT_VALIDATE_ON_MOT20_02'
    report={'decision':decision,'baseline':baseline,'best_policy':best,'n_segments':len(segs),'n_policies':len(rows)}
    (out/'decision.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    md=['# A45_03 Precision-gated Merge','', '## Decision','', '```text',decision,'```','','## Best policy',json.dumps(best,indent=2,sort_keys=True),'','## Baseline',json.dumps(baseline,indent=2,sort_keys=True)]
    (out/'decision.md').write_text('\n'.join(md)+'\n')
    print(json.dumps({'decision':decision,'best_HOTA':best.get('HOTA'),'best_policy':best.get('policy')},indent=2,sort_keys=True))
if __name__=='__main__': main()

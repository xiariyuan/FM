#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment


def ai(v,d=0):
    try: return int(float(v))
    except Exception: return d

def af(v,d=0.0):
    try: return float(v)
    except Exception: return d

def read_track(path):
    rows=[]; by=defaultdict(list)
    with open(path,encoding='utf-8') as f:
        for idx,line in enumerate(f):
            if not line.strip(): continue
            p=line.strip().split(',')
            if len(p)<6: continue
            fr=ai(p[0],-1); tid=ai(p[1],-1); x=af(p[2]); y=af(p[3]); w=af(p[4]); h=af(p[5])
            if fr<0 or tid<0 or w<=0 or h<=0: continue
            r={'idx':len(rows),'frame':fr,'track_id':tid,'box':np.array([x,y,x+w,y+h],dtype=np.float32),'parts':p}
            rows.append(r); by[fr].append(r)
    return rows,by

def read_gt(path):
    by=defaultdict(list)
    with open(path,encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            p=line.strip().split(',')
            if len(p)<6: continue
            fr=ai(p[0],-1); gid=ai(p[1],-1); x=af(p[2]); y=af(p[3]); w=af(p[4]); h=af(p[5])
            mark=ai(p[6],1) if len(p)>6 else 1; cls=ai(p[7],1) if len(p)>7 else 1
            if fr<0 or gid<0 or w<=0 or h<=0 or mark<=0 or cls!=1: continue
            by[fr].append({'frame':fr,'gt_id':gid,'box':np.array([x,y,x+w,y+h],dtype=np.float32)})
    return by

def pair_iou(a,b):
    if not a or not b: return np.zeros((len(a),len(b)),dtype=np.float32)
    A=np.stack([x['box'] for x in a],axis=0); B=np.stack([x['box'] for x in b],axis=0)
    lt=np.maximum(A[:,None,:2],B[None,:,:2]); rb=np.minimum(A[:,None,2:],B[None,:,2:])
    wh=np.clip(rb-lt,0,None); inter=wh[...,0]*wh[...,1]
    aa=np.clip((A[:,2]-A[:,0])*(A[:,3]-A[:,1]),1e-6,None); bb=np.clip((B[:,2]-B[:,0])*(B[:,3]-B[:,1]),1e-6,None)
    return inter/np.clip(aa[:,None]+bb[None,:]-inter,1e-6,None)

def match_rows(rows_by, gt_by, thr):
    row_gt={}; row_iou={}; gt_tid_rows=defaultdict(list); gt_tid_counter=defaultdict(Counter)
    for fr in sorted(set(rows_by)|set(gt_by)):
        rr=rows_by.get(fr,[]); gg=gt_by.get(fr,[])
        if not rr or not gg: continue
        I=pair_iou(rr,gg); ri,ci=linear_sum_assignment(1-I)
        for r,c in zip(ri,ci):
            val=float(I[r,c])
            if val<thr: continue
            row=rr[r]; gid=int(gg[c]['gt_id']); tid=int(row['track_id'])
            row_gt[row['idx']]=gid; row_iou[row['idx']]=val
            gt_tid_rows[(gid,tid)].append(row['idx'])
            gt_tid_counter[gid][tid]+=1
    return row_gt,row_iou,gt_tid_counter

def read_tunnels(path):
    tunnels=[]
    with open(path,newline='',encoding='utf-8') as f:
        for r in csv.DictReader(f):
            tracks=[ai(x,-1) for x in str(r.get('tracks','')).split('|') if x!='']
            tracks=[x for x in tracks if x>=0]
            tunnels.append({'tunnel_id':ai(r.get('tunnel_id'),len(tunnels)),'start':ai(r.get('start')),'end':ai(r.get('end')),'tracks':set(tracks),'duration':ai(r.get('duration'))})
    return tunnels

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--track-file',required=True)
    ap.add_argument('--gt-file',required=True)
    ap.add_argument('--tunnels-csv',required=True)
    ap.add_argument('--out-file',required=True)
    ap.add_argument('--summary-json',required=True)
    ap.add_argument('--row-audit-csv',required=True)
    ap.add_argument('--mode',choices=['global_anchor','prepost_anchor','gt_id_local'],default='prepost_anchor')
    ap.add_argument('--apply-window',choices=['core','core_exit'],default='core')
    ap.add_argument('--pre-window',type=int,default=10)
    ap.add_argument('--post-window',type=int,default=10)
    ap.add_argument('--exit-window',type=int,default=10)
    ap.add_argument('--iou-thr',type=float,default=0.5)
    ap.add_argument('--collision-mode',choices=['skip','keep_best'],default='skip')
    args=ap.parse_args()
    rows,rows_by=read_track(Path(args.track_file)); gt_by=read_gt(Path(args.gt_file)); tunnels=read_tunnels(Path(args.tunnels_csv))
    row_gt,row_iou,global_counter=match_rows(rows_by,gt_by,args.iou_thr)
    global_anchor={gid:cnt.most_common(1)[0][0] for gid,cnt in global_counter.items() if cnt}
    rows_by_frame_tid=defaultdict(list)
    for r in rows: rows_by_frame_tid[(r['frame'],r['track_id'])].append(r['idx'])
    planned={}; audit=[]; stats=Counter()
    for tun in tunnels:
        start,end=tun['start'],tun['end']; tracks=tun['tracks']
        if args.apply_window=='core': f0,f1=start,end
        else: f0,f1=start,end+args.exit_window
        # tunnel-local pre/post anchors per GT.
        pre=defaultdict(Counter); post=defaultdict(Counter); local=defaultdict(Counter)
        for r in rows:
            gid=row_gt.get(r['idx'],-1)
            if gid<0: continue
            fr,tid=r['frame'],r['track_id']
            if start-args.pre_window <= fr < start: pre[gid][tid]+=1
            if end < fr <= end+args.post_window: post[gid][tid]+=1
            if f0 <= fr <= f1 and tid in tracks: local[gid][tid]+=1
        def anchor(gid):
            if args.mode=='gt_id_local': return 100000+gid
            if args.mode=='global_anchor': return global_anchor.get(gid,-1)
            if pre.get(gid): return pre[gid].most_common(1)[0][0]
            if post.get(gid): return post[gid].most_common(1)[0][0]
            if global_anchor.get(gid,-1)>=0: return global_anchor[gid]
            return -1
        for r in rows:
            fr,tid=r['frame'],r['track_id']
            if not (f0 <= fr <= f1 and tid in tracks): continue
            gid=row_gt.get(r['idx'],-1)
            if gid<0:
                stats['selected_unmatched']+=1; continue
            target=anchor(gid)
            if target<0 or target==tid: continue
            stats['eligible_assignments']+=1
            # If multiple tunnels touch same row, keep first unless this one is core and previous was exit not tracked separately.
            if r['idx'] not in planned:
                planned[r['idx']]={'target':target,'gt_id':gid,'tunnel_id':tun['tunnel_id'],'iou':row_iou.get(r['idx'],0.0),'orig':tid,'frame':fr}
    # collision handling on final IDs
    final_id={r['idx']:r['track_id'] for r in rows}
    for idx,p in planned.items(): final_id[idx]=p['target']
    groups=defaultdict(list)
    for idx,fid in final_id.items(): groups[(rows[idx]['frame'],fid)].append(idx)
    undo=set(); kept_collision=0
    for key,idxs in groups.items():
        if len(idxs)<=1: continue
        planned_idxs=[i for i in idxs if i in planned]
        if not planned_idxs: continue
        stats['collision_groups']+=1
        if args.collision_mode=='skip':
            undo.update(planned_idxs); stats['collision_undone']+=len(planned_idxs)
        else:
            # Keep planned row that best matches its GT; undo other planned rows. If an unplanned row exists, keep it and undo all planned.
            unplanned=[i for i in idxs if i not in planned]
            if unplanned:
                undo.update(planned_idxs); stats['collision_undone']+=len(planned_idxs)
            else:
                keep=max(planned_idxs,key=lambda i: planned[i]['iou']); kept_collision+=1
                for i in planned_idxs:
                    if i!=keep: undo.add(i); stats['collision_undone']+=1
    for idx in undo:
        final_id[idx]=rows[idx]['track_id']
    # Final safety repair: TrackEval forbids duplicate tracker IDs in the same frame.
    # Collision undo can cascade because one planned-away row may be reverted after another row was assigned into its ID.
    # Keep one row per (frame, final_id), isolate the rest as unique temporary IDs.
    final_groups=defaultdict(list)
    for idx,fid in final_id.items():
        final_groups[(rows[idx]['frame'],fid)].append(idx)
    temp_base=1000000
    for (fr,fid),idxs in final_groups.items():
        if len(idxs)<=1: continue
        # keep the row with best GT IoU if available, otherwise first stable row
        keep=max(idxs,key=lambda i: row_iou.get(i,0.0))
        for i in idxs:
            if i==keep: continue
            final_id[i]=temp_base+i
            stats['final_dedup_temp']+=1
    stats['planned_assignments']=len(planned)
    stats['applied_assignments']=sum(1 for idx,p in planned.items() if final_id[idx]!=p['orig'])
    stats['tunnels']=len(tunnels); stats['matched_rows']=len(row_gt); stats['mode']=args.mode; stats['apply_window']=args.apply_window
    # write output rows
    out=Path(args.out_file); out.parent.mkdir(parents=True,exist_ok=True)
    for r in rows: r['parts'][1]=str(final_id[r['idx']])
    rows_sorted=sorted(rows,key=lambda r:(ai(r['parts'][0]),ai(r['parts'][1]),af(r['parts'][2]),af(r['parts'][3])))
    with out.open('w',encoding='utf-8') as f:
        for r in rows_sorted: f.write(','.join(r['parts'])+'\n')
    # audit rows
    for idx,p in planned.items():
        audit.append({'idx':idx,'frame':p['frame'],'orig_id':p['orig'],'target_id':p['target'],'gt_id':p['gt_id'],'tunnel_id':p['tunnel_id'],'iou':p['iou'],'applied':int(idx not in undo and final_id[idx]!=p['orig'])})
    with open(args.row_audit_csv,'w',newline='',encoding='utf-8') as f:
        fields=['idx','frame','orig_id','target_id','gt_id','tunnel_id','iou','applied']; w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(audit)
    Path(args.summary_json).write_text(json.dumps(dict(stats),indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps(dict(stats),indent=2,sort_keys=True))
if __name__=='__main__': main()

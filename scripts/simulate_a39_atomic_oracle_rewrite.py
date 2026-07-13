#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, json
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
    rows=[]; by=defaultdict(list)
    with path.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            p=line.strip().split(',')
            if len(p)<6:
                continue
            fr=ai(p[0],-1); tid=ai(p[1],-1)
            x=af(p[2]); y=af(p[3]); w=af(p[4]); h=af(p[5])
            if fr<0 or tid<0 or w<=0 or h<=0:
                continue
            r={'idx':len(rows),'frame':fr,'track_id':tid,'box':np.array([x,y,x+w,y+h],dtype=np.float32),'parts':p}
            rows.append(r); by[fr].append(r)
    return rows,by


def read_gt(path: Path):
    by=defaultdict(list)
    with path.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            p=line.strip().split(',')
            if len(p)<6:
                continue
            fr=ai(p[0],-1); gid=ai(p[1],-1)
            x=af(p[2]); y=af(p[3]); w=af(p[4]); h=af(p[5])
            mark=ai(p[6],1) if len(p)>6 else 1
            cls=ai(p[7],1) if len(p)>7 else 1
            if fr<0 or gid<0 or w<=0 or h<=0 or mark<=0 or cls!=1:
                continue
            by[fr].append({'frame':fr,'gt_id':gid,'box':np.array([x,y,x+w,y+h],dtype=np.float32)})
    return by


def pair_iou(a,b):
    if not a or not b:
        return np.zeros((len(a),len(b)),dtype=np.float32)
    A=np.stack([x['box'] for x in a]); B=np.stack([x['box'] for x in b])
    lt=np.maximum(A[:,None,:2],B[None,:,:2]); rb=np.minimum(A[:,None,2:],B[None,:,2:])
    wh=np.clip(rb-lt,0,None); inter=wh[...,0]*wh[...,1]
    aa=np.clip((A[:,2]-A[:,0])*(A[:,3]-A[:,1]),1e-6,None)
    bb=np.clip((B[:,2]-B[:,0])*(B[:,3]-B[:,1]),1e-6,None)
    return inter/np.clip(aa[:,None]+bb[None,:]-inter,1e-6,None)


def match_rows(rows_by, gt_by, thr):
    row_gt={}; row_iou={}; gt_tid_counter=defaultdict(Counter)
    for fr in sorted(set(rows_by)|set(gt_by)):
        rr=rows_by.get(fr,[]); gg=gt_by.get(fr,[])
        if not rr or not gg:
            continue
        I=pair_iou(rr,gg); ri,ci=linear_sum_assignment(1-I)
        for r,c in zip(ri,ci):
            val=float(I[r,c])
            if val<thr:
                continue
            row=rr[r]; gid=int(gg[c]['gt_id']); tid=int(row['track_id'])
            row_gt[row['idx']]=gid; row_iou[row['idx']]=val; gt_tid_counter[gid][tid]+=1
    return row_gt,row_iou,gt_tid_counter


def read_tunnels(path: Path):
    out=[]
    with path.open(newline='',encoding='utf-8') as f:
        for r in csv.DictReader(f):
            tracks=[ai(x,-1) for x in str(r.get('tracks','')).split('|') if x!='']
            tracks=[x for x in tracks if x>=0]
            out.append({'tunnel_id':ai(r.get('tunnel_id'),len(out)),'start':ai(r.get('start')),'end':ai(r.get('end')),'duration':ai(r.get('duration')),'tracks':set(tracks)})
    return out


def anchor_for(mode,gid,pre,post,global_anchor):
    if mode=='gt_id_local': return 100000+gid,'gt_id_local'
    if mode=='global_anchor': return global_anchor.get(gid,-1),'global_anchor'
    if pre.get(gid): return pre[gid].most_common(1)[0][0],'pre'
    if post.get(gid): return post[gid].most_common(1)[0][0],'post'
    return global_anchor.get(gid,-1),'global_fallback'


def write_csv(path,fields,rows):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)


def main():
    ap=argparse.ArgumentParser(description='A39_03x GT-guided tunnel-wide atomic rewrite engine smoke test')
    ap.add_argument('--track-file',required=True); ap.add_argument('--gt-file',required=True); ap.add_argument('--tunnels-csv',required=True)
    ap.add_argument('--out-file',required=True); ap.add_argument('--summary-json',required=True); ap.add_argument('--summary-md')
    ap.add_argument('--transaction-csv',required=True); ap.add_argument('--row-audit-csv',required=True); ap.add_argument('--collision-csv',required=True)
    ap.add_argument('--mode',choices=['prepost_anchor','global_anchor','gt_id_local'],default='prepost_anchor')
    ap.add_argument('--apply-window',choices=['core','core_exit'],default='core_exit')
    ap.add_argument('--pre-window',type=int,default=10); ap.add_argument('--post-window',type=int,default=10); ap.add_argument('--exit-window',type=int,default=10)
    ap.add_argument('--iou-thr',type=float,default=0.5)
    ap.add_argument('--collision-mode',choices=['skip','keep_best','rollback_transaction'],default='skip')
    args=ap.parse_args()

    rows,rows_by=read_track(Path(args.track_file)); gt_by=read_gt(Path(args.gt_file)); tunnels=read_tunnels(Path(args.tunnels_csv))
    row_gt,row_iou,global_counter=match_rows(rows_by,gt_by,args.iou_thr)
    global_anchor={gid:cnt.most_common(1)[0][0] for gid,cnt in global_counter.items() if cnt}
    stats=Counter(); txs=[]; tx_by_id={}; row_plan={}

    for tun in tunnels:
        start,end=tun['start'],tun['end']; tracks=tun['tracks']; f0=start; f1=end if args.apply_window=='core' else end+args.exit_window
        pre=defaultdict(Counter); post=defaultdict(Counter); source_by_gt=defaultdict(list); source_tracks=defaultdict(set)
        for r in rows:
            gid=row_gt.get(r['idx'],-1)
            if gid<0: continue
            fr=r['frame']; tid=r['track_id']
            if start-args.pre_window <= fr < start: pre[gid][tid]+=1
            if end < fr <= end+args.post_window: post[gid][tid]+=1
            if f0 <= fr <= f1 and tid in tracks:
                source_by_gt[gid].append(r['idx']); source_tracks[gid].add(tid)
        stats['tunnels_seen']+=1
        for gid,idxs0 in source_by_gt.items():
            target,anchor_source=anchor_for(args.mode,gid,pre,post,global_anchor)
            if target<0:
                stats['transactions_no_anchor']+=1; continue
            idxs=[i for i in idxs0 if rows[i]['track_id']!=target]
            if not idxs:
                stats['transactions_noop']+=1; continue
            tx_id=len(txs)
            tx={'transaction_id':tx_id,'tunnel_id':tun['tunnel_id'],'gt_id':gid,'target_id':target,'anchor_source':anchor_source,'frame_start':f0,'frame_end':f1,'planned_rows':len(idxs),'source_track_ids':sorted(source_tracks[gid]),'row_indices':idxs,'rolled_back':False}
            txs.append(tx); tx_by_id[tx_id]=tx
            for idx in idxs:
                if idx not in row_plan:
                    row_plan[idx]=tx_id

    orig={r['idx']:r['track_id'] for r in rows}; final=dict(orig)
    for idx,tx_id in row_plan.items(): final[idx]=tx_by_id[tx_id]['target_id']

    collision_rows=[]; undo=set(); rollback_txs=set(); groups=defaultdict(list)
    for idx,fid in final.items(): groups[(rows[idx]['frame'],fid)].append(idx)
    for (fr,fid),idxs in groups.items():
        if len(idxs)<=1: continue
        planned=[i for i in idxs if i in row_plan]
        if not planned: continue
        stats['collision_groups']+=1; stats['collision_rows']+=len(idxs)
        tx_ids=sorted({row_plan[i] for i in planned}); chosen=''
        if args.collision_mode=='skip':
            undo.update(planned); stats['collision_undone']+=len(planned)
        elif args.collision_mode=='rollback_transaction':
            rollback_txs.update(tx_ids)
        else:
            chosen=max(idxs,key=lambda i: row_iou.get(i,0.0))
            for i in planned:
                if i!=chosen:
                    undo.add(i); stats['collision_undone']+=1
        collision_rows.append({'frame':fr,'target_id':fid,'row_count':len(idxs),'planned_count':len(planned),'transaction_ids':'|'.join(map(str,tx_ids)),'chosen_idx':chosen,'mode':args.collision_mode})

    if rollback_txs:
        for tx_id in rollback_txs:
            tx_by_id[tx_id]['rolled_back']=True
            for idx in tx_by_id[tx_id]['row_indices']:
                if idx in row_plan: undo.add(idx)
        stats['rollback_transactions']=len(rollback_txs)
    for idx in undo: final[idx]=orig[idx]

    final_temp=set(); final_groups=defaultdict(list)
    for idx,fid in final.items(): final_groups[(rows[idx]['frame'],fid)].append(idx)
    temp_base=1000000
    for (_,fid),idxs in final_groups.items():
        if len(idxs)<=1: continue
        keep=max(idxs,key=lambda i: row_iou.get(i,0.0))
        for i in idxs:
            if i==keep: continue
            final[i]=temp_base+i; final_temp.add(i); stats['final_dedup_temp_rows']+=1

    tx_rows=[]
    for tx in txs:
        idxs=tx['row_indices']
        applied=sum(1 for i in idxs if final.get(i)!=orig[i] and i not in final_temp)
        undone=sum(1 for i in idxs if i in undo)
        temp=sum(1 for i in idxs if i in final_temp)
        if tx.get('rolled_back'): status='rollback_transaction'
        elif applied==0: status='no_effect'
        elif applied==len(idxs): status='committed_full'
        else: status='committed_partial'
        stats[f'tx_{status}']+=1
        tx_rows.append({'transaction_id':tx['transaction_id'],'tunnel_id':tx['tunnel_id'],'gt_id':tx['gt_id'],'target_id':tx['target_id'],'anchor_source':tx['anchor_source'],'source_track_ids':'|'.join(map(str,tx['source_track_ids'])),'frame_start':tx['frame_start'],'frame_end':tx['frame_end'],'planned_rows':tx['planned_rows'],'applied_rows':applied,'undone_rows':undone,'temp_rows':temp,'status':status})

    row_audit=[]
    for idx,tx_id in sorted(row_plan.items()):
        tx=tx_by_id[tx_id]
        if idx in final_temp: action='final_temp'
        elif idx in undo: action='undone'
        elif final[idx]!=orig[idx]: action='applied'
        else: action='noop'
        row_audit.append({'idx':idx,'frame':rows[idx]['frame'],'orig_id':orig[idx],'target_id':tx['target_id'],'final_id':final[idx],'gt_id':tx['gt_id'],'tunnel_id':tx['tunnel_id'],'transaction_id':tx_id,'iou':row_iou.get(idx,0.0),'action':action})

    for k,v in {'rows':len(rows),'matched_rows':len(row_gt),'tunnels':len(tunnels),'transactions':len(txs),'planned_rows':len(row_plan),'applied_rows':sum(1 for i in row_plan if final[i]!=orig[i] and i not in final_temp),'undone_rows':len(undo),'mode':args.mode,'apply_window':args.apply_window,'collision_mode':args.collision_mode,'pre_window':args.pre_window,'post_window':args.post_window,'exit_window':args.exit_window,'iou_thr':args.iou_thr}.items():
        stats[k]=v

    out=Path(args.out_file); out.parent.mkdir(parents=True,exist_ok=True)
    for r in rows: r['parts'][1]=str(final[r['idx']])
    with out.open('w',encoding='utf-8') as f:
        for r in sorted(rows,key=lambda r:(ai(r['parts'][0]),ai(r['parts'][1]),af(r['parts'][2]),af(r['parts'][3]))): f.write(','.join(r['parts'])+'\n')
    write_csv(args.transaction_csv,['transaction_id','tunnel_id','gt_id','target_id','anchor_source','source_track_ids','frame_start','frame_end','planned_rows','applied_rows','undone_rows','temp_rows','status'],tx_rows)
    write_csv(args.row_audit_csv,['idx','frame','orig_id','target_id','final_id','gt_id','tunnel_id','transaction_id','iou','action'],row_audit)
    write_csv(args.collision_csv,['frame','target_id','row_count','planned_count','transaction_ids','chosen_idx','mode'],collision_rows)
    Path(args.summary_json).write_text(json.dumps(dict(stats),indent=2,sort_keys=True)+'\n',encoding='utf-8')
    if args.summary_md:
        md=['# A39_03x GT-guided atomic rewrite engine','',f'mode: `{args.mode}`',f'apply_window: `{args.apply_window}`',f'collision_mode: `{args.collision_mode}`','','## Summary','```json',json.dumps(dict(stats),indent=2,sort_keys=True),'```']
        Path(args.summary_md).write_text('\n'.join(md)+'\n',encoding='utf-8')
    print(json.dumps(dict(stats),indent=2,sort_keys=True))


if __name__=='__main__':
    main()

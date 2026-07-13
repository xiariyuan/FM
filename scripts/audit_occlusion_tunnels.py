#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, math
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np


def read_track(path):
    by=defaultdict(list)
    with open(path, encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            p=line.strip().split(',')
            if len(p)<6: continue
            fr=int(float(p[0])); tid=int(float(p[1])); x=float(p[2]); y=float(p[3]); w=float(p[4]); h=float(p[5])
            if w<=0 or h<=0: continue
            by[fr].append({'frame':fr,'track_id':tid,'box':np.array([x,y,x+w,y+h],dtype=np.float32),'tlwh':(x,y,w,h)})
    return by


def pair_ioa(boxes):
    n=len(boxes)
    if n==0: return np.zeros((0,0),dtype=np.float32)
    a=np.stack(boxes,axis=0)
    lt=np.maximum(a[:,None,:2],a[None,:,:2]); rb=np.minimum(a[:,None,2:],a[None,:,2:])
    wh=np.clip(rb-lt,0,None); inter=wh[...,0]*wh[...,1]
    area=np.clip((a[:,2]-a[:,0])*(a[:,3]-a[:,1]),1e-6,None)
    den=np.minimum(area[:,None],area[None,:])
    out=inter/np.clip(den,1e-6,None)
    np.fill_diagonal(out,0.0)
    return out


def union_box(rows):
    b=np.stack([r['box'] for r in rows],axis=0)
    return np.array([b[:,0].min(),b[:,1].min(),b[:,2].max(),b[:,3].max()],dtype=np.float32)


def box_iou(a,b):
    lt=np.maximum(a[:2],b[:2]); rb=np.minimum(a[2:],b[2:]); wh=np.clip(rb-lt,0,None)
    inter=float(wh[0]*wh[1]); aa=float((a[2]-a[0])*(a[3]-a[1])); bb=float((b[2]-b[0])*(b[3]-b[1]))
    return inter/max(1e-6,aa+bb-inter)


def components_for_frame(rows, thr):
    n=len(rows)
    if n<2: return []
    ioa=pair_ioa([r['box'] for r in rows])
    parent=list(range(n))
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]]; x=parent[x]
        return x
    def union(a,b):
        ra,rb=find(a),find(b)
        if ra!=rb: parent[rb]=ra
    inds=np.argwhere(ioa>=thr)
    for i,j in inds:
        if i<j: union(int(i),int(j))
    comp=defaultdict(list)
    for i in range(n): comp[find(i)].append(i)
    out=[]
    for ids in comp.values():
        if len(ids)>=2:
            rr=[rows[i] for i in ids]
            tids=sorted(int(r['track_id']) for r in rr)
            vals=[]
            for a in range(len(ids)):
                for b in range(a+1,len(ids)):
                    vals.append(float(ioa[ids[a],ids[b]]))
            out.append({'tids':tids,'rows':rr,'max_ioa':max(vals) if vals else 0.0,'mean_ioa':sum(vals)/len(vals) if vals else 0.0,'ubox':union_box(rr)})
    return out


def load_events(path):
    ev=[]
    with open(path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            try:
                fr=int(float(r['frame'])); tid=int(float(r['chosen_tid']))
            except Exception:
                continue
            ev.append({'frame':fr,'track_id':tid,'is_gt_idsw':int(float(r.get('is_gt_idsw',0))),
                       'is_track_switch':int(float(r.get('is_track_switch',0))),
                       'is_bad_commit':int(float(r.get('is_bad_commit_before',0))),
                       'low_margin_005':int(float(r.get('low_margin_005',0))),
                       'det_gt':int(float(r.get('det_gt',-1)))})
    return ev


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--track-file',required=True)
    ap.add_argument('--events-csv',required=True)
    ap.add_argument('--out-dir',required=True)
    ap.add_argument('--ioa-thr',type=float,default=0.5)
    ap.add_argument('--min-duration',type=int,default=3)
    ap.add_argument('--max-gap',type=int,default=1)
    ap.add_argument('--min-shared-tracks',type=int,default=1)
    ap.add_argument('--union-iou-link',type=float,default=0.05)
    ap.add_argument('--exit-window',type=int,default=10)
    args=ap.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    by=read_track(args.track_file)
    frame_comps=[]; comp_id=0
    for fr in sorted(by):
        for c in components_for_frame(by[fr],args.ioa_thr):
            c.update({'component_id':comp_id,'frame':fr})
            frame_comps.append(c); comp_id+=1
    # link frame components into tunnels
    parent=list(range(len(frame_comps)))
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]]; x=parent[x]
        return x
    def union(a,b):
        ra,rb=find(a),find(b)
        if ra!=rb: parent[rb]=ra
    recent=[]
    for i,c in enumerate(frame_comps):
        fr=c['frame']; tids=set(c['tids'])
        recent=[j for j in recent if fr-frame_comps[j]['frame']<=args.max_gap]
        for j in recent:
            p=frame_comps[j]; shared=len(tids & set(p['tids']))
            if shared>=args.min_shared_tracks or box_iou(c['ubox'],p['ubox'])>=args.union_iou_link:
                union(i,j)
        recent.append(i)
    groups=defaultdict(list)
    for i,c in enumerate(frame_comps): groups[find(i)].append(i)
    tunnels=[]; comp_to_tunnel={}; frame_tid_to_tunnels=defaultdict(set)
    for tidx, ids in enumerate(groups.values()):
        comps=[frame_comps[i] for i in ids]
        frames=sorted({c['frame'] for c in comps})
        duration=frames[-1]-frames[0]+1
        if duration<args.min_duration: continue
        tids=sorted(set(t for c in comps for t in c['tids']))
        tunnel_id=len(tunnels)
        for i in ids: comp_to_tunnel[i]=tunnel_id
        for c in comps:
            for t in c['tids']: frame_tid_to_tunnels[(c['frame'],t)].add(tunnel_id)
        tunnels.append({'tunnel_id':tunnel_id,'start':frames[0],'end':frames[-1],'duration':duration,
                        'component_count':len(comps),'active_frame_count':len(frames),'track_count':len(tids),
                        'tracks':tids,'max_ioa':max(c['max_ioa'] for c in comps),'mean_ioa':sum(c['mean_ioa'] for c in comps)/len(comps)})
    # event coverage
    events=load_events(args.events_csv)
    totals=Counter(); covered=Counter(); exitcov=Counter(); in_or_exit=Counter()
    event_rows=[]
    # interval candidates for exit coverage
    tunnels_by_track=defaultdict(list)
    for t in tunnels:
        for tid in t['tracks']:
            tunnels_by_track[tid].append(t)
    for ev in events:
        kinds=[]
        if ev['is_gt_idsw']: kinds.append('gt_idsw')
        if ev['is_track_switch']: kinds.append('track_switch')
        if ev['is_bad_commit']: kinds.append('bad_commit')
        if ev['low_margin_005']: kinds.append('low_margin_005')
        kinds.append('all_events')
        in_ids=frame_tid_to_tunnels.get((ev['frame'],ev['track_id']),set())
        exit_ids=[]
        for t in tunnels_by_track.get(ev['track_id'],[]):
            if t['end'] < ev['frame'] <= t['end']+args.exit_window:
                exit_ids.append(t['tunnel_id'])
        for k in kinds:
            totals[k]+=1
            if in_ids: covered[k]+=1
            if exit_ids: exitcov[k]+=1
            if in_ids or exit_ids: in_or_exit[k]+=1
        if ev['is_gt_idsw'] or ev['is_track_switch'] or ev['is_bad_commit']:
            event_rows.append({**ev,'in_tunnel_ids':'|'.join(map(str,sorted(in_ids))), 'exit_tunnel_ids':'|'.join(map(str,sorted(exit_ids)))})
    def rate(a,b): return float(a)/float(b) if b else 0.0
    summary={'params':vars(args),'track_frames':len(by),'frame_components':len(frame_comps),'tunnels':len(tunnels),
             'totals':dict(totals),'covered_in_tunnel':dict(covered),'covered_exit_window':dict(exitcov),'covered_in_or_exit':dict(in_or_exit),
             'rates':{k:{'in_tunnel':rate(covered[k],totals[k]),'exit_window':rate(exitcov[k],totals[k]),'in_or_exit':rate(in_or_exit[k],totals[k])} for k in totals}}
    # write files
    with open(out/'frame_components.csv','w',newline='',encoding='utf-8') as f:
        fields=['component_id','frame','track_count','tracks','max_ioa','mean_ioa']
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for c in frame_comps:
            w.writerow({'component_id':c['component_id'],'frame':c['frame'],'track_count':len(c['tids']),'tracks':'|'.join(map(str,c['tids'])),'max_ioa':c['max_ioa'],'mean_ioa':c['mean_ioa']})
    with open(out/'tunnel_candidates.csv','w',newline='',encoding='utf-8') as f:
        fields=['tunnel_id','start','end','duration','component_count','active_frame_count','track_count','tracks','max_ioa','mean_ioa']
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for t in tunnels:
            row=t.copy(); row['tracks']='|'.join(map(str,row['tracks'])); w.writerow(row)
    with open(out/'error_events_tunnel_coverage.csv','w',newline='',encoding='utf-8') as f:
        fields=['frame','track_id','det_gt','is_gt_idsw','is_track_switch','is_bad_commit','low_margin_005','in_tunnel_ids','exit_tunnel_ids']
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(event_rows)
    (out/'tunnel_summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    md=['# A39 occlusion tunnel discovery audit','',f"params: ioa_thr={args.ioa_thr}, min_duration={args.min_duration}, max_gap={args.max_gap}, exit_window={args.exit_window}",'',f"frame_components: {len(frame_comps)}",f"tunnels: {len(tunnels)}",'', '| event | total | in_tunnel | rate | exit_window | rate | in_or_exit | rate |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for k in ['gt_idsw','track_switch','bad_commit','low_margin_005','all_events']:
        if totals[k]:
            md.append(f"| {k} | {totals[k]} | {covered[k]} | {rate(covered[k],totals[k]):.4f} | {exitcov[k]} | {rate(exitcov[k],totals[k]):.4f} | {in_or_exit[k]} | {rate(in_or_exit[k],totals[k]):.4f} |")
    (out/'tunnel_summary.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print('\n'.join(md))

if __name__=='__main__': main()

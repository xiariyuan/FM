#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def ai(v,d=0):
    try: return int(float(v))
    except Exception: return d


def af(v,d=0.0):
    try: return float(v)
    except Exception: return d


def read_csv(path):
    with Path(path).open(newline='',encoding='utf-8') as f:
        return list(csv.DictReader(f))


def read_track(path: Path):
    rows=[]; by_frame=defaultdict(list); by_frame_tid=defaultdict(list)
    with path.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            p=line.strip().split(',')
            if len(p)<6: continue
            fr=ai(p[0],-1); tid=ai(p[1],-1); x=af(p[2]); y=af(p[3]); w=af(p[4]); h=af(p[5])
            if fr<0 or tid<0 or w<=0 or h<=0: continue
            r={'idx':len(rows),'frame':fr,'track_id':tid,'box':np.array([x,y,x+w,y+h],dtype=np.float32),'parts':p}
            rows.append(r); by_frame[fr].append(r); by_frame_tid[(fr,tid)].append(r)
    return rows,by_frame,by_frame_tid


def read_gt(path: Path):
    by=defaultdict(list)
    with path.open(encoding='utf-8') as f:
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
            if val<thr: continue
            row_gt[rr[r]['idx']]=int(gg[c]['gt_id']); row_iou[rr[r]['idx']]=val
    return row_gt,row_iou


def read_tunnels(path):
    out={}
    with Path(path).open(newline='',encoding='utf-8') as f:
        for r in csv.DictReader(f):
            tracks=[ai(x,-1) for x in str(r.get('tracks','')).split('|') if x!='']; tracks=[x for x in tracks if x>=0]
            tid=ai(r.get('tunnel_id'),len(out)); out[tid]={'tunnel_id':tid,'start':ai(r.get('start')),'end':ai(r.get('end')),'tracks':set(tracks)}
    return out


POLICIES={
    'strict': {'sim':0.65,'row_margin':0.05,'col_margin':0.03},
    'normal': {'sim':0.60,'row_margin':0.04,'col_margin':0.02},
    'coverage': {'sim':0.55,'row_margin':0.03,'col_margin':0.01},
}


def accept(pair, p):
    return (af(pair['sim'])>=p['sim'] and af(pair['row_margin'])>=p['row_margin'] and af(pair['col_margin'])>=p['col_margin'] and ai(pair['pre_track'])!=ai(pair['post_track']))


def main():
    ap=argparse.ArgumentParser(description='A39_03a dry-run audit from existing A39_02b ReID Hungarian pair CSV')
    ap.add_argument('--pairs-csv',required=True)
    ap.add_argument('--groups-csv',required=True)
    ap.add_argument('--track-file',required=True)
    ap.add_argument('--gt-file',required=True)
    ap.add_argument('--tunnels-csv',required=True)
    ap.add_argument('--out-dir',required=True)
    ap.add_argument('--iou-thr',type=float,default=0.5)
    ap.add_argument('--exit-window',type=int,default=10)
    args=ap.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)

    pairs=read_csv(args.pairs_csv); groups=read_csv(args.groups_csv); tunnels=read_tunnels(args.tunnels_csv)
    rows,rows_by,by_frame_tid=read_track(Path(args.track_file)); gt_by=read_gt(Path(args.gt_file)); row_gt,row_iou=match_rows(rows_by,gt_by,args.iou_thr)
    gmap={}
    for g in groups:
        key=(ai(g['tunnel_id']),g['side'],ai(g['track_id']))
        gmap[key]={'major_gt':ai(g.get('major_gt'),-1),'purity':af(g.get('gt_purity'),0.0),'rows':ai(g.get('rows'),0),'crops':ai(g.get('crops'),0)}

    audit=[]; forecast=[]; summary=[]
    for pair in pairs:
        tid=ai(pair['tunnel_id']); pre_id=ai(pair['pre_track']); post_id=ai(pair['post_track'])
        pre_g=gmap.get((tid,'pre',pre_id),{}); post_g=gmap.get((tid,'post',post_id),{})
        pre_gt=pre_g.get('major_gt',-1); post_gt=post_g.get('major_gt',-1); gt_same=int(pre_gt>=0 and pre_gt==post_gt)
        row={'tunnel_id':tid,'pre_track':pre_id,'post_track':post_id,'sim':af(pair['sim']),'row_margin':af(pair['row_margin']),'col_margin':af(pair['col_margin']),'pre_gt':pre_gt,'post_gt':post_gt,'gt_same':gt_same,'pre_purity':pre_g.get('purity',0.0),'post_purity':post_g.get('purity',0.0),'pre_rows':pre_g.get('rows',0),'post_rows':post_g.get('rows',0)}
        for name,pol in POLICIES.items(): row[f'accept_{name}']=int(accept(pair,pol))
        audit.append(row)

    for policy,pol in POLICIES.items():
        st=Counter(); wrong=[]
        for a in audit:
            if not a[f'accept_{policy}']: continue
            st['accepted']+=1; st['gt_same']+=a['gt_same']; st['wrong']+=1-a['gt_same']
            if not a['gt_same']: wrong.append(a)
            tun=tunnels.get(a['tunnel_id'])
            if not tun: continue
            f0,f1=tun['start'],tun['end']+args.exit_window
            target=a['pre_track']; post=a['post_track']; gid=a['post_gt']
            # A39_02b-style minimal forecast: only post-track rows after tunnel exit.
            post_rows=[r for fr in range(tun['end']+1,tun['end']+args.exit_window+1) for r in by_frame_tid.get((fr,post),[])]
            # Tunnel-wide oracle diagnostic forecast: if assignment is GT-correct, rows of same GT inside core+exit.
            oracle_rows=[]
            if a['gt_same'] and gid>=0:
                for r in rows:
                    if f0 <= r['frame'] <= f1 and r['track_id'] in tun['tracks'] and row_gt.get(r['idx'],-1)==gid and r['track_id']!=target:
                        oracle_rows.append(r)
            collision_post=sum(1 for r in post_rows if by_frame_tid.get((r['frame'],target)))
            collision_oracle=sum(1 for r in oracle_rows if by_frame_tid.get((r['frame'],target)))
            st['post_rows']+=len(post_rows); st['post_collision_rows']+=collision_post
            st['oracle_rows']+=len(oracle_rows); st['oracle_collision_rows']+=collision_oracle
            forecast.append({'policy':policy,'tunnel_id':a['tunnel_id'],'pre_track':target,'post_track':post,'gt_same':a['gt_same'],'sim':a['sim'],'row_margin':a['row_margin'],'col_margin':a['col_margin'],'post_rows':len(post_rows),'post_collision_rows':collision_post,'oracle_core_exit_rows':len(oracle_rows),'oracle_collision_rows':collision_oracle})
        precision=st['gt_same']/st['accepted'] if st['accepted'] else 0.0
        summary.append({'policy':policy,'accepted':st['accepted'],'gt_same':st['gt_same'],'wrong':st['wrong'],'precision':precision,'post_rows':st['post_rows'],'post_collision_rows':st['post_collision_rows'],'oracle_core_exit_rows':st['oracle_rows'],'oracle_collision_rows':st['oracle_collision_rows']})

    def wc(path,fields,records):
        with (out/path).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(records)
    wc('assignment_audit.csv',['tunnel_id','pre_track','post_track','sim','row_margin','col_margin','pre_gt','post_gt','gt_same','pre_purity','post_purity','pre_rows','post_rows','accept_strict','accept_normal','accept_coverage'],audit)
    wc('transaction_forecast.csv',['policy','tunnel_id','pre_track','post_track','gt_same','sim','row_margin','col_margin','post_rows','post_collision_rows','oracle_core_exit_rows','oracle_collision_rows'],forecast)
    wc('policy_summary.csv',['policy','accepted','gt_same','wrong','precision','post_rows','post_collision_rows','oracle_core_exit_rows','oracle_collision_rows'],summary)
    payload={'policies':POLICIES,'summary':summary,'inputs':vars(args)}
    (out/'policy_summary.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    md=['# A39_03a assignment dry-run from existing ReID pairs','','| policy | accepted | gt_same | wrong | precision | post_rows | post_collision_rows | oracle_core_exit_rows | oracle_collision_rows |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        md.append(f"| {r['policy']} | {r['accepted']} | {r['gt_same']} | {r['wrong']} | {r['precision']:.4f} | {r['post_rows']} | {r['post_collision_rows']} | {r['oracle_core_exit_rows']} | {r['oracle_collision_rows']} |")
    (out/'policy_summary.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print('\n'.join(md))


if __name__=='__main__': main()

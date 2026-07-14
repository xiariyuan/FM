from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import networkx as nx
import pandas as pd

SEQS=['MOT20-01','MOT20-02','MOT20-03','MOT20-05']


def read_csv(path:Path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def read_tracks(path:Path):
    rows=[];spans={}
    with path.open() as f:
        for line in f:
            p=line.strip().split(',')
            if len(p)<6: continue
            fr=int(float(p[0]));tid=int(float(p[1]));rows.append(p)
            if tid not in spans:spans[tid]=[fr,fr]
            else:spans[tid]=[min(spans[tid][0],fr),max(spans[tid][1],fr)]
    return rows,spans


def add_ranks(df:pd.DataFrame,col:str):
    x=df.sort_values(['seq','track_a',col,'gap','track_b'],ascending=[True,True,False,True,True]).copy()
    x['out_rank']=x.groupby(['seq','track_a']).cumcount()+1
    x=x.sort_values(['seq','track_b',col,'gap','track_a'],ascending=[True,True,False,True,True])
    x['in_rank']=x.groupby(['seq','track_b']).cumcount()+1
    x['max_rank_policy']=x[['out_rank','in_rank']].max(axis=1)
    return x


def select_matching(edges:List[dict],weight_mode:str,locked_origin:str|None=None):
    selected=[]
    used_left=set();used_right=set()
    if locked_origin:
        locked=sorted([e for e in edges if e['origin']==locked_origin],key=lambda r:-float(r['base_weight']))
        parent={}
        def find(x):
            parent.setdefault(x,x)
            while parent[x]!=x:
                parent[x]=parent[parent[x]];x=parent[x]
            return x
        for e in locked:
            a=int(float(e['track_a']));b=int(float(e['track_b']))
            if a in used_left or b in used_right:continue
            ra,rb=find(a),find(b)
            if ra==rb:continue
            parent[rb]=ra;used_left.add(a);used_right.add(b);selected.append(e)
    graph=nx.Graph()
    pair_to_edge={}
    for e in edges:
        if locked_origin and e['origin']==locked_origin:continue
        a=int(float(e['track_a']));b=int(float(e['track_b']))
        if a in used_left or b in used_right:continue
        left=('L',a);right=('R',b)
        if weight_mode=='raw':w=float(e['base_weight'])
        elif weight_mode=='margin':w=max(1e-9,float(e['score_pct'])-float(e['cutoff']))/max(1e-9,1-float(e['cutoff']))
        elif weight_mode=='hybrid':w=max(0,float(e['base_weight']))+max(0,float(e['score_pct'])-float(e['cutoff']))/max(1e-9,1-float(e['cutoff']))
        else:raise ValueError(weight_mode)
        if e['origin']=='a42':w+=float(e.get('origin_boost',0))
        if e['origin']=='adaptive':w+=float(e.get('origin_boost',0))
        if w<=0:continue
        key=(left,right)
        old=pair_to_edge.get(key)
        if old is None or w>old[0]:pair_to_edge[key]=(w,e)
    for (left,right),(w,e) in pair_to_edge.items():
        graph.add_edge(left,right,weight=w)
    match=nx.algorithms.matching.max_weight_matching(graph,maxcardinality=False,weight='weight')
    for u,v in match:
        if u[0]=='R':u,v=v,u
        selected.append(pair_to_edge[(u,v)][1])
    return selected


def apply_edges(src:Path,out:Path,edges:List[dict]):
    rows,spans=read_tracks(src);parent={};accepted=[]
    def find(x):
        parent.setdefault(x,x)
        while parent[x]!=x:
            parent[x]=parent[parent[x]];x=parent[x]
        return x
    for e in sorted(edges,key=lambda r:(int(float(r['start_b'])),int(float(r['end_a'])))):
        a=int(float(e['track_a']));b=int(float(e['track_b']))
        if a not in spans or b not in spans or spans[a][1]>=spans[b][0]:continue
        ra,rb=find(a),find(b)
        if ra==rb:continue
        parent[rb]=ra;accepted.append(e)
    involved={int(float(e[k])) for e in accepted for k in ['track_a','track_b']}
    idmap={t:find(t) for t in involved}
    frames=defaultdict(list);outrows=[];changed=0
    for p in rows:
        q=list(p);tid=int(float(q[1]));new=idmap.get(tid,tid);changed+=int(new!=tid);q[1]=str(new)
        frames[int(float(q[0]))].append(new);outrows.append(q)
    dup=[fr for fr,ids in frames.items() if len(ids)!=len(set(ids))]
    if dup:raise RuntimeError(f'duplicate IDs {src}: {dup[:5]}')
    out.parent.mkdir(parents=True,exist_ok=True);outrows.sort(key=lambda p:(int(float(p[0])),int(float(p[1])),float(p[2]),float(p[3])))
    with out.open('w') as f:
        for p in outrows:f.write(','.join(p)+'\n')
    return accepted,{'links':len(accepted),'changed_rows':changed,'a42':sum(e['origin']=='a42' for e in accepted),'adaptive':sum(e['origin']=='adaptive' for e in accepted)}


def evaluate(pdir:Path,name:str):
    cmd=[sys.executable,'scripts/eval_motstyle_trackeval.py','--benchmark-name','MOT20','--split-to-eval','train','--gt-root','datasets/MOT20/train','--results-dir',str(pdir/'track_results'),'--tracker-name',name,'--work-dir',str(pdir/'eval_work'),'--seqs',*SEQS]
    p=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);(pdir/'eval.log').write_text(p.stdout)
    detail=pdir/'eval_work/eval'/name/'pedestrian_detailed.csv';res={'returncode':p.returncode}
    if detail.exists():
        rows=list(csv.DictReader(detail.open()))
        for seq in SEQS+['COMBINED']:
            r=next((x for x in rows if x['seq']==seq),None)
            if r:res[seq]={'HOTA':float(r['HOTA___AUC'])*100,'DetA':float(r['DetA___AUC'])*100,'AssA':float(r['AssA___AUC'])*100,'IDF1':float(r['IDF1'])*100 if float(r['IDF1'])<2 else float(r['IDF1']),'IDSW':int(float(r['IDSW']))}
        res['simple_avg_HOTA']=sum(res[s]['HOTA'] for s in SEQS)/4
    return res


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--scores',required=True);ap.add_argument('--diagnostics',required=True);ap.add_argument('--a42-links',required=True);ap.add_argument('--source-dir',required=True);ap.add_argument('--out-dir',required=True);args=ap.parse_args()
    col='meta_risk_score_l1p0';use=['seq','track_a','track_b','gap','start_b','end_a','endpoint_reid_aligned','appearance_max','max_debt_pct','assa_merge_delta_proxy','assa_merge_positive',col,col+'_seqpct']
    df=pd.read_csv(args.scores,usecols=lambda c:c in set(use));ranked=add_ranks(df,col);diag=pd.read_csv(args.diagnostics);a42=read_csv(Path(args.a42_links));out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    adaptive_by_seq={}
    for seq in SEQS:
        row=diag[(diag.score_col==col)&(diag.max_rank==2)&(diag.aggregation=='q75')&(diag.heldout_seq==seq)].iloc[0];cut=float(row.learned_cutoff)
        s=ranked[(ranked.seq==seq)&(ranked.max_rank_policy<=2)&(ranked.endpoint_reid_aligned==1)&(ranked.appearance_max>=.65)&(ranked.max_debt_pct>=.5)&(ranked[col+'_seqpct']>=cut)]
        rows=[]
        for r in s.to_dict('records'):
            r.update({'origin':'adaptive','base_weight':float(r[col]),'score_pct':float(r[col+'_seqpct']),'cutoff':cut,'origin_boost':0.0});rows.append(r)
        adaptive_by_seq[seq]=rows
    a42_by_seq=defaultdict(list)
    # Add current spans from score table where available; otherwise infer from source tracks below.
    for r in a42:
        x=dict(r);x.update({'origin':'a42','base_weight':float(r['a42_model_score']),'score_pct':1.0,'cutoff':0.0,'origin_boost':0.0});a42_by_seq[r['seq']].append(x)
    policies=[
      {'name':'adaptive_mwm_raw','include_a42':False,'weight_mode':'raw','locked':None,'a42_boost':0},
      {'name':'adaptive_mwm_margin','include_a42':False,'weight_mode':'margin','locked':None,'a42_boost':0},
      {'name':'joint_mwm_hybrid_a42b1','include_a42':True,'weight_mode':'hybrid','locked':None,'a42_boost':1},
      {'name':'joint_mwm_hybrid_a42b3','include_a42':True,'weight_mode':'hybrid','locked':None,'a42_boost':3},
      {'name':'a42_locked_adaptive_mwm','include_a42':True,'weight_mode':'hybrid','locked':'a42','a42_boost':0},
    ]
    summaries=[]
    for pol in policies:
        pdir=out/pol['name'];tdir=pdir/'track_results';tdir.mkdir(parents=True,exist_ok=True);allsel=[];byseq=[]
        for seq in SEQS:
            _,spans=read_tracks(Path(args.source_dir)/f'{seq}.txt')
            edges=[dict(e) for e in adaptive_by_seq[seq]]
            if pol['include_a42']:
                for e0 in a42_by_seq[seq]:
                    e=dict(e0);a=int(float(e['track_a']));b=int(float(e['track_b']))
                    if a not in spans or b not in spans:continue
                    e['end_a']=spans[a][1];e['start_b']=spans[b][0];e['origin_boost']=pol['a42_boost'];edges.append(e)
            for e in edges:
                if 'end_a' not in e:
                    a=int(float(e['track_a']));b=int(float(e['track_b']));e['end_a']=spans[a][1];e['start_b']=spans[b][0]
            chosen=select_matching(edges,pol['weight_mode'],pol['locked'])
            accepted,st=apply_edges(Path(args.source_dir)/f'{seq}.txt',tdir/f'{seq}.txt',chosen);st.update({'seq':seq,'candidates':len(edges),'chosen_matching':len(chosen),'true_utility':sum(float(e.get('assa_merge_delta_proxy',0) or 0) for e in accepted),'positive':sum(float(e.get('assa_merge_delta_proxy',0) or 0)>0 for e in accepted)});byseq.append(st);allsel+=accepted
        fields=[]
        for r in allsel:
            for k in r:
                if k not in fields:fields.append(k)
        with (pdir/'selected_links.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(allsel)
        ev=evaluate(pdir,'mwm_'+pol['name']);summary={'policy':pol,'selected':len(allsel),'a42':sum(r['origin']=='a42' for r in allsel),'adaptive':sum(r['origin']=='adaptive' for r in allsel),'by_seq':byseq,'eval':ev}
        (pdir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');summaries.append(summary)
        print(json.dumps({'name':pol['name'],'selected':summary['selected'],'origins':[summary['a42'],summary['adaptive']],'by_seq':byseq,'combined':ev.get('COMBINED'),'m02':ev.get('MOT20-02'),'avg':ev.get('simple_avg_HOTA')},indent=2),flush=True)
    (out/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
if __name__=='__main__':main()

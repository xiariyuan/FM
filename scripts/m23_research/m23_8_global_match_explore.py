from __future__ import annotations

# Research artifact for the MOT20 M23 sequence-LOSO audit.
import csv, json, math, subprocess, sys
from collections import defaultdict
from pathlib import Path
import pandas as pd
import networkx as nx

SEQS=['MOT20-01','MOT20-02','MOT20-03','MOT20-05']
PARENT=Path('outputs/assocriskbench_p15_20260714/fused_a42_adaptive_risk_eval/l1_r2_q75_adaptive_priority/track_results')
SCORES=Path('outputs/spot_runtime_gate_20260628/A42_long_gap_global_association/A42_02b_ranking_model_train_eval/a42_train_oof_scores.csv')
OUT=Path('outputs/mot20_m23_20260718/global_match_explore_v1')

def read_tracks(path):
    rows=[]; spans={}
    with path.open() as f:
        for line in f:
            p=line.rstrip('\n').split(',')
            if len(p)<6: continue
            fr=int(float(p[0])); tid=int(float(p[1])); rows.append(p)
            spans.setdefault(tid,[fr,fr]); spans[tid][0]=min(spans[tid][0],fr); spans[tid][1]=max(spans[tid][1],fr)
    return rows,spans

def select_matching(g, spans, threshold):
    x=g[(g.a42_model_score>=threshold)&(g.appearance_max>=.55)&(g.geometry_risk<=1)&(g.motion_risk<=2)&(g.max_rank_by_a42_model_score<=3)].copy()
    x=x[x.track_a.isin(spans)&x.track_b.isin(spans)]
    x=x[[spans[int(a)][1]<spans[int(b)][0] for a,b in zip(x.track_a,x.track_b)]]
    G=nx.Graph()
    lookup={}
    for r in x.itertuples():
        a=int(r.track_a); b=int(r.track_b)
        # Logit rewards calibrated confidence; appearance breaks near-ties only.
        p=min(max(float(r.a42_model_score),1e-6),1-1e-6)
        w=math.log(p/(1-p))+0.01*float(r.appearance_max)
        u=('L',a); v=('R',b)
        if (u,v) not in lookup or w>lookup[(u,v)][0]: lookup[(u,v)]=(w,r)
    for (u,v),(w,r) in lookup.items(): G.add_edge(u,v,weight=w)
    matching=nx.algorithms.matching.max_weight_matching(G,maxcardinality=False,weight='weight')
    out=[]
    for u,v in matching:
        if u[0]=='R': u,v=v,u
        out.append(lookup[(u,v)][1])
    return sorted(out,key=lambda r:(-float(r.a42_model_score),int(r.gap),int(r.track_a),int(r.track_b)))

def apply(rows,spans,edges):
    parent={}
    def find(x):
        parent.setdefault(x,x)
        while parent[x]!=x:
            parent[x]=parent[parent[x]];x=parent[x]
        return x
    selected=[]
    for r in edges:
        a=int(r.track_a); b=int(r.track_b); ra=find(a); rb=find(b)
        if ra==rb: continue
        parent[rb]=ra; selected.append(r)
    ids={int(r.track_a) for r in selected}|{int(r.track_b) for r in selected}
    idmap={t:find(t) for t in ids}
    out=[]; frames=defaultdict(list); changed=0
    for p in rows:
        q=list(p); old=int(float(q[1])); new=idmap.get(old,old); q[1]=str(new); changed+=new!=old
        frames[int(float(q[0]))].append(new); out.append(q)
    dup=[f for f,ids2 in frames.items() if len(ids2)!=len(set(ids2))]
    if dup: raise RuntimeError(f'duplicate frame IDs: {dup[:10]}')
    out.sort(key=lambda p:(int(float(p[0])),int(float(p[1])),float(p[2]),float(p[3])))
    return out,selected,changed

def evaluate(root,name):
    cmd=[sys.executable,'scripts/eval_motstyle_trackeval.py','--benchmark-name','MOT20','--split-to-eval','train','--gt-root','datasets/MOT20/train','--results-dir',str(root/'track_results'),'--tracker-name',name,'--work-dir',str(root/'eval_work'),'--seqs',*SEQS]
    p=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (root/'eval.log').write_text(p.stdout)
    if p.returncode: raise RuntimeError(p.stdout[-4000:])
    detail=root/'eval_work/eval'/name/'pedestrian_detailed.csv'
    rr=list(csv.DictReader(detail.open())); out={}
    for s in SEQS+['COMBINED']:
        r=next(x for x in rr if x['seq']==s)
        out[s]={'HOTA':100*float(r['HOTA___AUC']),'DetA':100*float(r['DetA___AUC']),'AssA':100*float(r['AssA___AUC']),'IDSW':int(float(r['IDSW']))}
    return out

def main():
    df=pd.read_csv(SCORES)
    OUT.mkdir(parents=True,exist_ok=True)
    summaries=[]
    for threshold in [.80,.90,.95]:
        name=f'global_match_t{int(threshold*100):02d}'
        root=OUT/name; td=root/'track_results'; td.mkdir(parents=True,exist_ok=True)
        allsel=[]; by=[]
        for seq in SEQS:
            rows,spans=read_tracks(PARENT/f'{seq}.txt')
            edges=select_matching(df[df.seq.eq(seq)],spans,threshold)
            out,sel,changed=apply(rows,spans,edges)
            with (td/f'{seq}.txt').open('w') as f:
                for p in out:f.write(','.join(p)+'\n')
            for r in sel:
                d=r._asdict(); d['seq']=seq; allsel.append(d)
            by.append({'seq':seq,'eligible_matching_edges':len(edges),'selected':len(sel),'changed_rows':int(changed),'diagnostic_tp':int(sum(int(r.same_gt) for r in sel))})
        if allsel:
            with (root/'selected_links.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(allsel[0]));w.writeheader();w.writerows(allsel)
        ev=evaluate(root,name)
        s={'threshold':threshold,'by_seq':by,'selected_total':len(allsel),'diagnostic_tp':int(sum(int(r['same_gt']) for r in allsel)),'eval':ev}
        (root/'summary.json').write_text(json.dumps(s,indent=2)+'\n'); summaries.append(s)
        print(json.dumps({'threshold':threshold,'selected':len(allsel),'tp':s['diagnostic_tp'],'combined':ev['COMBINED'],'m02':ev['MOT20-02'],'m05':ev['MOT20-05']},indent=2),flush=True)
    (OUT/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
if __name__=='__main__':main()

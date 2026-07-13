#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, json, shutil, subprocess, sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import networkx as nx

TRAIN_SEQS = ["MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05"]


def af(v,d=0.0):
    try:
        if v is None or v=='': return d
        return float(v)
    except Exception: return d

def ai(v,d=0):
    try:
        if v is None or v=='': return d
        return int(float(v))
    except Exception: return d

def read_csv(p: Path):
    with p.open(newline='',encoding='utf-8') as f: return list(csv.DictReader(f))

def write_csv(p: Path, rows):
    rows=list(rows); fields=[]; seen=set()
    for r in rows:
        for k in r:
            if k not in seen: seen.add(k); fields.append(k)
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields or ['seq'],extrasaction='ignore'); w.writeheader(); w.writerows(rows)

def read_mot(path: Path):
    out=[]
    with path.open('r',encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if not line: continue
            parts=line.split(',')
            if len(parts)<6: continue
            out.append((int(float(parts[0])), int(float(parts[1])), parts))
    return out

def find(parent: Dict[int,int], x:int)->int:
    parent.setdefault(x,x)
    while parent[x]!=x:
        parent[x]=parent[parent[x]]; x=parent[x]
    return x

def gate_pass(row,cfg):
    if af(row.get('aflink_score')) < float(cfg.get('min_aflink',-1)): return False
    if 'min_adjusted' in cfg and af(row.get('debt_adjusted_edge_score')) < float(cfg['min_adjusted']): return False
    if af(row.get('out_rank_by_aflink_score'),999) > float(cfg.get('max_out_rank',999)): return False
    if af(row.get('in_rank_by_aflink_score'),999) > float(cfg.get('max_in_rank',999)): return False
    if min(af(row.get('out_margin_to_second_aflink_score')),af(row.get('in_margin_to_second_aflink_score'))) < float(cfg.get('min_bidir_margin',-999)): return False
    if af(row.get('edge_debt_score')) < float(cfg.get('min_debt',0)): return False
    if af(row.get('risk_total')) > float(cfg.get('max_risk',999)): return False
    if af(row.get('geometry_risk')) > float(cfg.get('max_geometry_risk',999)): return False
    if af(row.get('motion_risk')) > float(cfg.get('max_motion_risk',999)): return False
    if af(row.get('competition_risk')) > float(cfg.get('max_competition_risk',999)): return False
    es=cfg.get('edge_set')
    if es == 'no_highrisk' and row.get('edge_type') == 'high_risk_geometry': return False
    if es == 'boundary_or_fragment' and row.get('edge_type') not in {'weak_boundary_recovery','fragmented_tracklet_recovery'}: return False
    if es == 'stable_gap' and row.get('edge_type') not in {'short_gap_continuation','long_gap_reappearance'}: return False
    return ai(row.get('gap'),-1) > 0

def run_trackeval(linked_dir: Path, tracker_name: str, out_root: Path) -> Tuple[Path, dict]:
    eval_root=out_root/'trackeval'
    tracker_data=eval_root/'trackers'/tracker_name/'data'
    tracker_data.mkdir(parents=True,exist_ok=True)
    for seq in TRAIN_SEQS:
        shutil.copy2(linked_dir/f'{seq}.txt', tracker_data/f'{seq}.txt')
    seqmap_dir=eval_root/'seqmaps'; seqmap_dir.mkdir(parents=True,exist_ok=True)
    seqmap=seqmap_dir/'MOT20_train.txt'
    seqmap.write_text('name\n'+'\n'.join(TRAIN_SEQS)+'\n',encoding='utf-8')
    cmd=[sys.executable,'TrackEval/scripts/run_mot_challenge.py','--GT_FOLDER','datasets/MOT20/train','--TRACKERS_FOLDER',str(eval_root/'trackers'),'--OUTPUT_FOLDER',str(eval_root/'eval'),'--TRACKERS_TO_EVAL',tracker_name,'--BENCHMARK','MOT20','--SPLIT_TO_EVAL','train','--SEQMAP_FILE',str(seqmap),'--SKIP_SPLIT_FOL','True','--DO_PREPROC','True','--TRACKER_SUB_FOLDER','data','--OUTPUT_SUB_FOLDER','','--PRINT_ONLY_COMBINED','True','--METRICS','HOTA','CLEAR','Identity']
    p=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (out_root/'trackeval_stdout.log').write_text(p.stdout,encoding='utf-8')
    summary=eval_root/'eval'/tracker_name/'pedestrian_summary.txt'
    metrics={}
    if summary.exists():
        lines=[x.strip() for x in summary.read_text().splitlines() if x.strip()]
        if len(lines)>=2: metrics=dict(zip(lines[0].split(), lines[1].split()))
    metrics['trackeval_returncode']=p.returncode
    metrics['summary_path']=str(summary)
    return summary, metrics

def solve(policy,cfg,candidates,source_dir:Path,out_root:Path):
    pdir=out_root/policy; linked=pdir/'linked_results'; linked.mkdir(parents=True,exist_ok=True)
    gated=[r for r in candidates if gate_pass(r,cfg)]
    by=defaultdict(list)
    for r in gated: by[r['seq']].append(r)
    selected_all=[]; by_seq=[]; audit=[]
    for seq in TRAIN_SEQS:
        cand=by.get(seq,[])
        G=nx.Graph(); payload={}
        for i,r in enumerate(cand):
            a=ai(r['track_a']); b=ai(r['track_b'])
            w=af(r.get('debt_adjusted_edge_score')) + 1e-9*af(r.get('aflink_score')) - 1e-12*i
            if w <= 0: continue
            u=('src',a); v=('dst',b)
            G.add_edge(u,v,weight=w); payload[(u,v)]=r; payload[(v,u)]=r
        matching=nx.algorithms.matching.max_weight_matching(G,maxcardinality=False,weight='weight')
        matched=[]
        for u,v in matching:
            r=payload.get((u,v))
            if r is not None: matched.append(r)
        matched.sort(key=lambda r:(af(r.get('debt_adjusted_edge_score')),af(r.get('aflink_score'))),reverse=True)
        parent={}; used_s=set(); used_t=set(); final=[]
        for r in matched:
            a=ai(r['track_a']); b=ai(r['track_b'])
            if a in used_s or b in used_t: continue
            ra,rb=find(parent,a),find(parent,b)
            if ra==rb: continue
            parent[rb]=ra; used_s.add(a); used_t.add(b)
            q=dict(r); q['policy']=policy; q['selected_rank_in_seq']=len(final)+1; q['edge_weight']=af(r.get('debt_adjusted_edge_score'))
            final.append(q)
        ids=set()
        for r in final: ids.add(ai(r['track_a'])); ids.add(ai(r['track_b']))
        idmap={tid:find(parent,tid) for tid in ids}
        src=source_dir/f'{seq}.txt'; out=linked/f'{seq}.txt'
        rows=[]
        for _,tid,parts in read_mot(src):
            pp=list(parts); pp[1]=str(idmap.get(tid,tid)); rows.append(pp)
        rows.sort(key=lambda p:(int(float(p[0])), int(float(p[1])), float(p[2]), float(p[3])))
        with out.open('w',encoding='utf-8') as f:
            for pp in rows: f.write(','.join(pp)+'\n')
        selected_all.extend(final)
        tp=sum(str(r.get('same_gt'))=='1' for r in final)
        by_seq.append({'seq':seq,'gated_candidates':len(cand),'graph_edges':G.number_of_edges(),'accepted_links':len(final),'tp_train_label':tp,'precision_train_label':tp/len(final) if final else 0.0})
        audit.append({'seq':seq,'source_rows':sum(1 for _ in open(src)),'linked_rows':sum(1 for _ in open(out)),'row_count_ok':int(sum(1 for _ in open(src))==sum(1 for _ in open(out)))})
    write_csv(pdir/'accepted_links.csv',selected_all)
    summary_path, metrics=run_trackeval(linked, 'A41_02_train_'+policy, pdir)
    summary={'policy':policy,'gate_config':cfg,'gated_candidates_total':len(gated),'accepted_links_total':len(selected_all),'by_seq':by_seq,'input_output_audit':audit,'metrics':metrics,'decision':'PASS_EVAL_DONE' if metrics.get('trackeval_returncode')==0 else 'EVAL_FAILED'}
    (pdir/'link_summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    return summary

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--candidates',required=True)
    ap.add_argument('--gate-configs',required=True)
    ap.add_argument('--source-dir',required=True)
    ap.add_argument('--out-dir',required=True)
    ap.add_argument('--policies',nargs='*',default=['strict_p80','balanced_p70','aggressive_p60'])
    args=ap.parse_args()
    cand=read_csv(Path(args.candidates)); cfgs=json.loads(Path(args.gate_configs).read_text())
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    sums=[]
    for pol in args.policies:
        sums.append(solve(pol,cfgs[pol],cand,Path(args.source_dir),out))
    (out/'a41_02_train_eval_summary.json').write_text(json.dumps({'policies':sums,'decision':'A41_02_TRAIN_EVAL_DONE'},indent=2,sort_keys=True)+'\n')
    # compact csv
    fields=['policy','HOTA','IDF1','MOTA','AssA','DetA','IDSW','Frag','accepted_links_total','gated_candidates_total','decision','summary_path']
    with (out/'a41_02_train_eval_metrics.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for s in sums:
            m=s['metrics']
            w.writerow({k:(s.get(k) if k in s else m.get(k,'')) for k in fields})
    print(json.dumps([{'policy':s['policy'],'accepted':s['accepted_links_total'],'decision':s['decision'],'metrics':{k:s['metrics'].get(k) for k in ['HOTA','IDF1','MOTA','AssA','DetA','IDSW','Frag']}} for s in sums],indent=2,sort_keys=True))
if __name__=='__main__': main()

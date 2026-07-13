#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,shutil,subprocess,sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple, List, Set

TRAIN_SEQS=['MOT20-01','MOT20-02','MOT20-03','MOT20-05']

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

def read_csv(p:Path):
    with p.open(newline='',encoding='utf-8') as f: return list(csv.DictReader(f))
def write_csv(p:Path, rows):
    rows=list(rows); fields=[]; seen=set()
    for r in rows:
        for k in r:
            if k not in seen: seen.add(k); fields.append(k)
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields or ['seq'],extrasaction='ignore'); w.writeheader(); w.writerows(rows)
def key(row)->Tuple[str,str,str]: return (row.get('seq',''), str(int(float(row.get('track_a',0) or 0))), str(int(float(row.get('track_b',0) or 0))))
def key_s(k): return f'{k[0]}:{k[1]}->{k[2]}'
def parse_key(s):
    seq,rest=s.split(':',1); a,b=rest.split('->',1); return (seq,a,b)
def read_mot(path:Path):
    out=[]
    with path.open('r',encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if not line: continue
            parts=line.split(',')
            if len(parts)>=6: out.append((int(float(parts[0])),int(float(parts[1])),parts))
    return out
def find(parent:Dict[int,int],x:int)->int:
    parent.setdefault(x,x)
    while parent[x]!=x:
        parent[x]=parent[parent[x]]; x=parent[x]
    return x

def link_with_edges(edge_rows:List[dict], source_dir:Path, linked_dir:Path):
    linked_dir.mkdir(parents=True,exist_ok=True)
    by=defaultdict(list)
    for r in edge_rows: by[r['seq']].append(r)
    selected_all=[]; by_seq=[]
    for seq in TRAIN_SEQS:
        rows=sorted(by.get(seq,[]), key=lambda r:(af(r.get('debt_adjusted_edge_score')),af(r.get('aflink_score'))), reverse=True)
        parent={}; used_s=set(); used_t=set(); final=[]
        # Edge set is already teacher/A41 matching-compatible, but still guard one-to-one/cycles.
        for r in rows:
            a=ai(r['track_a']); b=ai(r['track_b'])
            if a in used_s or b in used_t: continue
            ra,rb=find(parent,a),find(parent,b)
            if ra==rb: continue
            parent[rb]=ra; used_s.add(a); used_t.add(b); final.append(r)
        ids=set()
        for r in final: ids.add(ai(r['track_a'])); ids.add(ai(r['track_b']))
        idmap={tid:find(parent,tid) for tid in ids}
        src=source_dir/f'{seq}.txt'; out=linked_dir/f'{seq}.txt'
        mot=[]
        for _,tid,parts in read_mot(src):
            pp=list(parts); pp[1]=str(idmap.get(tid,tid)); mot.append(pp)
        mot.sort(key=lambda p:(int(float(p[0])), int(float(p[1])), float(p[2]), float(p[3])))
        with out.open('w',encoding='utf-8') as f:
            for pp in mot: f.write(','.join(pp)+'\n')
        selected_all.extend(final)
        tp=sum(str(r.get('same_gt'))=='1' for r in final)
        by_seq.append({'seq':seq,'candidate_edges':len(rows),'accepted_links':len(final),'tp_train_label':tp,'precision_train_label':tp/len(final) if final else 0.0})
    return selected_all, by_seq

def interpolate(input_dir:Path, output_dir:Path, summary_json:Path, summary_csv:Path):
    cmd=[sys.executable,'scripts/postprocess/linear_interpolate_mot.py','--input-dir',str(input_dir),'--output-dir',str(output_dir),'--max-gap','30','--summary-json',str(summary_json),'--summary-csv',str(summary_csv)]
    p=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (output_dir.parent/'interp_stdout.json').write_text(p.stdout,encoding='utf-8')
    return p.returncode

def run_trackeval(track_dir:Path, tracker_name:str, out_dir:Path):
    eval_root=out_dir/'eval_mot20_all_train'
    data=eval_root/'trackers'/tracker_name/'data'; data.mkdir(parents=True,exist_ok=True)
    for seq in TRAIN_SEQS: shutil.copy2(track_dir/f'{seq}.txt', data/f'{seq}.txt')
    seqmap=eval_root/'seqmaps'/'MOT20_train.txt'; seqmap.parent.mkdir(parents=True,exist_ok=True); seqmap.write_text('name\n'+'\n'.join(TRAIN_SEQS)+'\n')
    cmd=[sys.executable,'TrackEval/scripts/run_mot_challenge.py','--GT_FOLDER','datasets/MOT20/train','--TRACKERS_FOLDER',str(eval_root/'trackers'),'--OUTPUT_FOLDER',str(eval_root/'eval'),'--TRACKERS_TO_EVAL',tracker_name,'--BENCHMARK','MOT20','--SPLIT_TO_EVAL','train','--SEQMAP_FILE',str(seqmap),'--SKIP_SPLIT_FOL','True','--DO_PREPROC','True','--TRACKER_SUB_FOLDER','data','--OUTPUT_SUB_FOLDER','','--PRINT_ONLY_COMBINED','True','--METRICS','HOTA','CLEAR','Identity']
    p=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (out_dir/'trackeval_stdout.log').write_text(p.stdout,encoding='utf-8')
    summary=eval_root/'eval'/tracker_name/'pedestrian_summary.txt'
    metrics={}
    if summary.exists():
        lines=[x.strip() for x in summary.read_text().splitlines() if x.strip()]
        if len(lines)>=2: metrics=dict(zip(lines[0].split(),lines[1].split()))
    metrics['trackeval_returncode']=p.returncode; metrics['summary_path']=str(summary)
    return metrics

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--features',required=True)
    ap.add_argument('--base-links',required=True)
    ap.add_argument('--policy-candidates',required=True)
    ap.add_argument('--source-dir',required=True)
    ap.add_argument('--out-dir',required=True)
    args=ap.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    features=read_csv(Path(args.features)); feat={key(r):r for r in features}
    base=read_csv(Path(args.base_links)); base_keys={key(r) for r in base}
    policies=json.load(open(args.policy_candidates))['policy_candidates']
    # Deduplicate by rule while keeping desired baselines.
    chosen=[]; seen=set()
    wanted=['union_score0.20_app0.60_risk2_rank1','union_score0.15_app0.60_risk4_rank1','union_score0.15_app0.60_risk7_rank2','p12_only_all','p14_only_all','teacher_any_all']
    for w in wanted:
        for p in policies:
            if p['rule']==w and w not in seen:
                chosen.append(p); seen.add(w)
    # Fallback to first candidate list if exact names missing.
    if not chosen: chosen=policies[:6]
    summaries=[]
    for p in chosen:
        rec_keys={parse_key(s) for s in p.get('selected_keys','').split('|') if s}
        union_keys=base_keys | rec_keys
        edge_rows=[]
        for k in sorted(union_keys):
            r=dict(feat[k]); r['hybrid_policy']=p['rule']; r['is_base_a41']=int(k in base_keys); r['is_recovery']=int(k in rec_keys and k not in base_keys)
            edge_rows.append(r)
        safe_name='hybrid_'+p['rule'].replace('.','p').replace('>','').replace('<','').replace(':','_')
        pdir=out/safe_name; raw_link=pdir/'raw_linked_results'; interp_dir=pdir/'track_results'
        selected,by_seq=link_with_edges(edge_rows,Path(args.source_dir),raw_link)
        write_csv(pdir/'accepted_links.csv',selected)
        interp_rc=interpolate(raw_link,interp_dir,pdir/'interp_summary.json',pdir/'interp_summary.csv')
        metrics=run_trackeval(interp_dir,'A41_05_'+safe_name,pdir)
        tp=sum(str(r.get('same_gt'))=='1' for r in selected)
        rec_tp=sum(str(r.get('same_gt'))=='1' and r.get('is_recovery')==1 for r in selected)
        summary={'policy':p['rule'],'safe_name':safe_name,'base_links':len(base_keys),'recovery_requested':len(rec_keys),'accepted_links_total':len(selected),'accepted_recovery_links':sum(int(r.get('is_recovery',0)) for r in selected),'tp_train_label':tp,'precision_train_label':tp/len(selected) if selected else 0.0,'recovery_tp_train_label':rec_tp,'by_seq':by_seq,'interp_returncode':interp_rc,'metrics':metrics,'decision':'PASS_EVAL_DONE' if metrics.get('trackeval_returncode')==0 else 'EVAL_FAILED'}
        (pdir/'link_summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
        summaries.append(summary)
    with (out/'a41_05b_summary.json').open('w',encoding='utf-8') as f: json.dump({'policies':summaries,'decision':'A41_05b_HYBRID_EVAL_DONE'},f,indent=2,sort_keys=True)
    with (out/'a41_05b_metrics.csv').open('w',newline='',encoding='utf-8') as f:
        fields=['policy','safe_name','accepted_links_total','accepted_recovery_links','tp_train_label','precision_train_label','HOTA','IDF1','MOTA','AssA','DetA','IDSW','Frag','decision','summary_path']
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for s in summaries:
            m=s['metrics']; w.writerow({'policy':s['policy'],'safe_name':s['safe_name'],'accepted_links_total':s['accepted_links_total'],'accepted_recovery_links':s['accepted_recovery_links'],'tp_train_label':s['tp_train_label'],'precision_train_label':s['precision_train_label'],**{k:m.get(k,'') for k in ['HOTA','IDF1','MOTA','AssA','DetA','IDSW','Frag']},'decision':s['decision'],'summary_path':m.get('summary_path','')})
    print(json.dumps([{ 'policy':s['policy'], 'accepted':s['accepted_links_total'], 'recovery':s['accepted_recovery_links'], 'metrics':{k:s['metrics'].get(k) for k in ['HOTA','IDF1','MOTA','AssA','IDSW','Frag']}, 'decision':s['decision']} for s in summaries],indent=2,sort_keys=True))
if __name__=='__main__': main()

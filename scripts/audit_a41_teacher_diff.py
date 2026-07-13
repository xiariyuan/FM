#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json
from collections import Counter,defaultdict
from pathlib import Path

def read_csv(p):
    with open(p,newline='',encoding='utf-8') as f: return list(csv.DictReader(f))
def write_csv(p,rows):
    rows=list(rows); fields=[]; seen=set()
    for r in rows:
        for k in r:
            if k not in seen: seen.add(k); fields.append(k)
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields or ['subset'],extrasaction='ignore'); w.writeheader(); w.writerows(rows)
def key(r): return (r.get('seq'), str(int(float(r.get('track_a')))), str(int(float(r.get('track_b')))))
def af(v,d=0.0):
    try: return float(v)
    except Exception: return d
def ai(v,d=0):
    try: return int(float(v))
    except Exception: return d

def summarize(name, keys, feat):
    rows=[]
    tp=0
    vals=defaultdict(list); c=Counter()
    for k in keys:
        r=feat.get(k,{})
        if not r: continue
        rows.append(r)
        tp += int(str(r.get('same_gt'))=='1')
        for col in ['aflink_score','debt_adjusted_edge_score','edge_debt_score','risk_total','geometry_risk','motion_risk','competition_risk','out_rank_by_aflink_score','in_rank_by_aflink_score','out_margin_to_second_aflink_score','in_margin_to_second_aflink_score','appearance_max']:
            vals[col].append(af(r.get(col)))
        c['edge_type:'+r.get('edge_type','')]+=1
        for tag in (r.get('source_debt_tags','')+'|'+r.get('target_debt_tags','')).split('|'):
            if tag: c['debt:'+tag]+=1
    out={'subset':name,'count':len(keys),'feature_found':len(rows),'tp':tp,'precision':tp/len(rows) if rows else 0.0}
    for col,vs in vals.items():
        if vs:
            out[col+'_mean']=sum(vs)/len(vs); out[col+'_min']=min(vs); out[col+'_max']=max(vs)
    out['top_counts']=json.dumps(c.most_common(20),ensure_ascii=False)
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--teacher-links', required=True)
    ap.add_argument('--student-links', required=True)
    ap.add_argument('--features', required=True)
    ap.add_argument('--teacher-name', default='teacher')
    ap.add_argument('--student-name', default='student')
    ap.add_argument('--out-dir', required=True)
    args=ap.parse_args()
    teacher=read_csv(args.teacher_links); student=read_csv(args.student_links); features=read_csv(args.features)
    feat={key(r):r for r in features}
    tk={key(r) for r in teacher}; sk={key(r) for r in student}
    subsets={
        'common': tk & sk,
        f'{args.teacher_name}_only': tk - sk,
        f'{args.student_name}_only': sk - tk,
        args.teacher_name: tk,
        args.student_name: sk,
    }
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    summary=[]
    for name,ks in subsets.items():
        rows=[feat[k] for k in sorted(ks) if k in feat]
        write_csv(out/(name+'.csv'), rows)
        summary.append(summarize(name,ks,feat))
    write_csv(out/'teacher_diff_summary.csv', summary)
    js={'teacher':args.teacher_name,'student':args.student_name,'teacher_links':len(tk),'student_links':len(sk),'summary':summary,'decision':'A41_04_TEACHER_DIFF_AUDIT_DONE'}
    (out/'teacher_diff_summary.json').write_text(json.dumps(js,indent=2,sort_keys=True)+'\n')
    print(json.dumps(js,indent=2,sort_keys=True))
if __name__=='__main__': main()

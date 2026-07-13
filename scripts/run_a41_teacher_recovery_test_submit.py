#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,shutil,subprocess,sys,zipfile,hashlib
from pathlib import Path
from typing import Dict,Tuple,List,Set
from collections import defaultdict

TEST_SEQS=['MOT20-04','MOT20-06','MOT20-07','MOT20-08']

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
def write_csv(p:Path,rows):
    rows=list(rows); fields=[]; seen=set()
    for r in rows:
        for k in r:
            if k not in seen: seen.add(k); fields.append(k)
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields or ['seq'],extrasaction='ignore'); w.writeheader(); w.writerows(rows)
def key(r)->Tuple[str,str,str]: return (r.get('seq',''), str(int(float(r.get('track_a',0) or 0))), str(int(float(r.get('track_b',0) or 0))))
def key_s(k): return f'{k[0]}:{k[1]}->{k[2]}'
def read_mot(p:Path):
    out=[]
    with p.open('r',encoding='utf-8') as f:
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
def source_file(source_by_seq_dir:Path, seq:str)->Path:
    p=source_by_seq_dir/seq/f'{seq}.txt'
    if p.exists(): return p
    p=source_by_seq_dir/f'{seq}.txt'
    if p.exists(): return p
    raise FileNotFoundError(seq)
def md5(p:Path)->str:
    h=hashlib.md5()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def line_count(p:Path)->int:
    with p.open('r',encoding='utf-8') as f: return sum(1 for _ in f)

def recovery_rule(r:dict, rule:str)->bool:
    # Current best train rule: union_score0.15_app0.60_risk7_rank2.
    if rule == 'union_score0.15_app0.60_risk7_rank2':
        max_rank=max(ai(r.get('out_rank_by_aflink_score'),999), ai(r.get('in_rank_by_aflink_score'),999))
        return (af(r.get('aflink_score'))>=0.15 and af(r.get('appearance_max'))>=0.60 and ai(r.get('risk_total'))<=7 and max_rank<=2 and ai(r.get('geometry_risk'))<=1 and ai(r.get('motion_risk'))<=2)
    if rule == 'union_score0.15_app0.60_risk4_rank1':
        max_rank=max(ai(r.get('out_rank_by_aflink_score'),999), ai(r.get('in_rank_by_aflink_score'),999))
        return (af(r.get('aflink_score'))>=0.15 and af(r.get('appearance_max'))>=0.60 and ai(r.get('risk_total'))<=4 and max_rank<=1 and ai(r.get('geometry_risk'))<=1 and ai(r.get('motion_risk'))<=2)
    if rule == 'union_score0.20_app0.60_risk2_rank1':
        max_rank=max(ai(r.get('out_rank_by_aflink_score'),999), ai(r.get('in_rank_by_aflink_score'),999))
        return (af(r.get('aflink_score'))>=0.20 and af(r.get('appearance_max'))>=0.60 and ai(r.get('risk_total'))<=2 and max_rank<=1 and ai(r.get('geometry_risk'))<=1 and ai(r.get('motion_risk'))<=2)
    raise KeyError(rule)

def link_with_edges(edge_rows:List[dict], source_by_seq_dir:Path, linked_dir:Path):
    linked_dir.mkdir(parents=True,exist_ok=True)
    by=defaultdict(list)
    for r in edge_rows: by[r['seq']].append(r)
    selected_all=[]; by_seq=[]; audit=[]
    for seq in TEST_SEQS:
        rows=sorted(by.get(seq,[]), key=lambda r:(af(r.get('debt_adjusted_edge_score')), af(r.get('aflink_score'))), reverse=True)
        parent={}; used_s=set(); used_t=set(); final=[]
        for r in rows:
            a=ai(r['track_a']); b=ai(r['track_b'])
            if a in used_s or b in used_t: continue
            ra,rb=find(parent,a),find(parent,b)
            if ra==rb: continue
            parent[rb]=ra; used_s.add(a); used_t.add(b); final.append(r)
        ids=set()
        for r in final: ids.add(ai(r['track_a'])); ids.add(ai(r['track_b']))
        idmap={tid:find(parent,tid) for tid in ids}
        src=source_file(source_by_seq_dir,seq); out=linked_dir/f'{seq}.txt'
        mot=[]
        for _,tid,parts in read_mot(src):
            pp=list(parts); pp[1]=str(idmap.get(tid,tid)); mot.append(pp)
        mot.sort(key=lambda p:(int(float(p[0])),int(float(p[1])),float(p[2]),float(p[3])))
        with out.open('w',encoding='utf-8') as f:
            for pp in mot: f.write(','.join(pp)+'\n')
        selected_all.extend(final)
        by_seq.append({'seq':seq,'candidate_edges':len(rows),'accepted_links':len(final),'recovery_links':sum(ai(r.get('is_recovery')) for r in final),'base_links':sum(ai(r.get('is_base_a41')) for r in final)})
        audit.append({'seq':seq,'source_file':str(src),'source_rows':line_count(src),'source_md5':md5(src),'linked_file':str(out),'linked_rows':line_count(out),'linked_md5':md5(out),'row_count_ok':int(line_count(src)==line_count(out))})
    return selected_all,by_seq,audit

def interpolate(input_dir:Path, output_dir:Path, summary_json:Path, summary_csv:Path):
    cmd=[sys.executable,'scripts/postprocess/linear_interpolate_mot.py','--input-dir',str(input_dir),'--output-dir',str(output_dir),'--max-gap','30','--summary-json',str(summary_json),'--summary-csv',str(summary_csv)]
    p=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (output_dir.parent/'interp_stdout.json').write_text(p.stdout,encoding='utf-8')
    return p.returncode

def package_and_validate(result_dir:Path, package_root:Path, zip_path:Path, out_dir:Path):
    package_root.mkdir(parents=True,exist_ok=True)
    for seq in TEST_SEQS: shutil.copy2(result_dir/f'{seq}.txt', package_root/f'{seq}.txt')
    if zip_path.exists(): zip_path.unlink()
    with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as zf:
        for seq in TEST_SEQS: zf.write(package_root/f'{seq}.txt', arcname=f'{seq}.txt')
    logs={}
    for name,cmd in {
        'results_dir':[sys.executable,'scripts/check_mot20_submission.py','--results-dir',str(package_root),'--profile','mot20_test_4'],
        'zip_path':[sys.executable,'scripts/check_mot20_submission.py','--zip-path',str(zip_path),'--profile','mot20_test_4'],
    }.items():
        p=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        log=out_dir/f'validation_{name}.txt'; log.write_text(p.stdout,encoding='utf-8')
        logs[name]={'returncode':p.returncode,'log_path':str(log)}
    return logs

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--features',required=True)
    ap.add_argument('--base-links',required=True)
    ap.add_argument('--teacher-links',nargs='+',required=True)
    ap.add_argument('--source-by-seq-dir',required=True)
    ap.add_argument('--out-dir',required=True)
    ap.add_argument('--rule',default='union_score0.15_app0.60_risk7_rank2')
    args=ap.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    features=read_csv(Path(args.features)); feat={key(r):r for r in features}
    base=read_csv(Path(args.base_links)); base_keys={key(r) for r in base}
    teacher_keys=set()
    for p in args.teacher_links:
        teacher_keys |= {key(r) for r in read_csv(Path(p))}
    recovery_keys={k for k in teacher_keys-base_keys if k in feat and recovery_rule(feat[k],args.rule)}
    all_keys=base_keys|recovery_keys
    edge_rows=[]
    for k in sorted(all_keys):
        r=dict(feat[k]); r['hybrid_rule']=args.rule; r['is_base_a41']=int(k in base_keys); r['is_recovery']=int(k in recovery_keys); r['teacher_union_member']=int(k in teacher_keys)
        edge_rows.append(r)
    raw_link=out/'raw_linked_results'; track_results=out/'track_results'
    selected,by_seq,audit=link_with_edges(edge_rows,Path(args.source_by_seq_dir),raw_link)
    write_csv(out/'accepted_links.csv',selected)
    interp_rc=interpolate(raw_link,track_results,out/'interp_summary.json',out/'interp_summary.csv')
    validation=package_and_validate(track_results,out/'package_root',out/f'MOT20_A41_05c_{args.rule}_submission.zip',out)
    summary={'rule':args.rule,'base_links':len(base_keys),'teacher_union_links':len(teacher_keys),'recovery_candidate_links':len(recovery_keys),'accepted_links_total':len(selected),'accepted_recovery_links':sum(ai(r.get('is_recovery')) for r in selected),'by_seq':by_seq,'input_output_audit':audit,'interp_returncode':interp_rc,'zip_path':str(out/f'MOT20_A41_05c_{args.rule}_submission.zip'),'validation':validation,'decision':'PASS_FORMAT_READY' if all(a['row_count_ok'] for a in audit) and interp_rc==0 and all(v['returncode']==0 for v in validation.values()) else 'CHECK_FAILED'}
    (out/'submission_summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    print(json.dumps(summary,indent=2,sort_keys=True))
if __name__=='__main__': main()

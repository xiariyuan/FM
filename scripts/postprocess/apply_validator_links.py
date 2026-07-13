#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


def ii(x, default=0):
    try: return int(float(x))
    except Exception: return default

def ff(x, default=0.0):
    try: return float(x)
    except Exception: return default


def find(parent, x):
    parent.setdefault(x, x)
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def read_edges(pred_csv: Path, seq: str, topk: int) -> list[tuple[int,int,float]]:
    rows=[]
    with pred_csv.open('r', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r.get('seq') != seq:
                continue
            rows.append((ii(r.get('track_a')), ii(r.get('track_b')), ff(r.get('validator_score')), ff(r.get('gap'))))
    rows.sort(key=lambda x: (-x[2], x[3], x[0], x[1]))
    selected=[]; used_a=set(); used_b=set(); parent={}
    for a,b,score,gap in rows:
        if a in used_a or b in used_b: continue
        ra=find(parent,a); rb=find(parent,b)
        if ra==rb: continue
        parent[rb]=ra; used_a.add(a); used_b.add(b)
        selected.append((a,b,score))
        if len(selected)>=topk: break
    return selected


def apply_links(input_file: Path, output_file: Path, edges: list[tuple[int,int,float]]) -> dict:
    parent={}
    for a,b,_ in edges:
        ra=find(parent,a); rb=find(parent,b); parent[rb]=ra
    involved=set()
    for a,b,_ in edges:
        involved.add(a); involved.add(b)
    id_map={tid:find(parent,tid) for tid in involved}
    rows=[]; remap=0
    with input_file.open('r', encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if not line: continue
            p=line.split(',')
            tid=ii(p[1])
            if tid in id_map and id_map[tid]!=tid:
                p[1]=str(id_map[tid]); remap+=1
            rows.append(p)
    rows.sort(key=lambda p:(ii(p[0]),ii(p[1]),ff(p[2]),ff(p[3])))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open('w', encoding='utf-8') as f:
        for p in rows: f.write(','.join(p)+'\n')
    return {'links':len(edges),'remapped_rows':remap,'rows':len(rows)}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input-dir', required=True)
    ap.add_argument('--pred-csv', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--seq-topk', action='append', default=[], help='SEQ:K')
    ap.add_argument('--copy-source', action='append', default=[], help='SEQ:/path/to/source.txt')
    ap.add_argument('--summary-json', default='')
    args=ap.parse_args()
    out=Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    summary=[]
    for spec in args.seq_topk:
        seq,k=spec.split(':',1); k=int(k)
        edges=read_edges(Path(args.pred_csv), seq, k)
        stats=apply_links(Path(args.input_dir)/f'{seq}.txt', out/f'{seq}.txt', edges)
        summary.append({'seq':seq,'mode':'validator','topk':k,**stats})
    for spec in args.copy_source:
        seq,src=spec.split(':',1)
        srcp=Path(src); dst=out/f'{seq}.txt'
        if not srcp.is_file(): raise FileNotFoundError(srcp)
        shutil.copy2(srcp,dst)
        summary.append({'seq':seq,'mode':'copy','source':str(srcp)})
    if args.summary_json:
        Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_json).write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    print(json.dumps(summary,indent=2,sort_keys=True))

if __name__=='__main__': main()

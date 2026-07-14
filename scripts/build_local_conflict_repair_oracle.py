from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class Edge:
    frame: int
    gt_id: int
    prev_tid: int
    new_tid: int
    overlap: float
    gap: int


@dataclass
class Component:
    frame: int
    index: int
    tids: Tuple[int, ...]
    edges: List[Edge]
    support: int = 0
    score: float = 0.0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_events(path: Path, seq: str) -> List[Edge]:
    out: List[Edge] = []
    with path.open(newline='') as f:
        for r in csv.DictReader(f):
            if r['seq'] != seq:
                continue
            out.append(Edge(
                frame=int(r['frame']),
                gt_id=int(r['gt_id']),
                prev_tid=int(r['prev_track_id']),
                new_tid=int(r['new_track_id']),
                overlap=float(r.get('either_overlap_max_ioa_1') or 0.0),
                gap=int(r.get('gap_since_prev_match') or 0),
            ))
    return out


def load_matches(path: Path, seq: str) -> Dict[Tuple[int, int], Tuple[int, float]]:
    # (frame, gt_id) -> (track_id, iou)
    out: Dict[Tuple[int, int], Tuple[int, float]] = {}
    with path.open(newline='') as f:
        for r in csv.DictReader(f):
            if r['seq'] != seq:
                continue
            out[(int(r['frame']), int(r['gt_id']))] = (int(r['track_id']), float(r['iou']))
    return out


def build_components(events: Sequence[Edge]) -> List[Component]:
    by_frame: Dict[int, List[Edge]] = defaultdict(list)
    for e in events:
        by_frame[e.frame].append(e)
    comps: List[Component] = []
    idx = 0
    for frame in sorted(by_frame):
        edges = by_frame[frame]
        adj: Dict[int, set[int]] = defaultdict(set)
        edge_by_tid: Dict[int, List[Edge]] = defaultdict(list)
        for e in edges:
            adj[e.prev_tid].add(e.new_tid)
            adj[e.new_tid].add(e.prev_tid)
            edge_by_tid[e.prev_tid].append(e)
            edge_by_tid[e.new_tid].append(e)
        seen: set[int] = set()
        for start in sorted(adj):
            if start in seen:
                continue
            q = deque([start]); seen.add(start); tids=[]; component_edges=set()
            while q:
                u=q.popleft(); tids.append(u)
                for e in edge_by_tid[u]:
                    component_edges.add(e)
                for v in adj[u]:
                    if v not in seen:
                        seen.add(v); q.append(v)
            comps.append(Component(frame=frame,index=idx,tids=tuple(sorted(tids)),edges=sorted(component_edges,key=lambda x:(x.gt_id,x.prev_tid,x.new_tid))))
            idx += 1
    return comps


def next_switch_frames(events: Sequence[Edge]) -> Dict[Tuple[int, int], int | None]:
    by_gt: Dict[int, List[int]] = defaultdict(list)
    for e in events:
        by_gt[e.gt_id].append(e.frame)
    out: Dict[Tuple[int, int], int | None] = {}
    for gt, frames in by_gt.items():
        frames=sorted(set(frames))
        for i, fr in enumerate(frames):
            out[(gt,fr)] = frames[i+1] if i+1 < len(frames) else None
    return out


def score_components(comps: List[Component], events: Sequence[Edge], matches: Dict[Tuple[int,int],Tuple[int,float]], max_horizon: int) -> None:
    nxt=next_switch_frames(events)
    for c in comps:
        support=0
        overlap_sum=0.0
        for e in c.edges:
            stop=nxt.get((e.gt_id,e.frame))
            if stop is None:
                stop=e.frame+max_horizon+1
            else:
                stop=min(stop,e.frame+max_horizon+1)
            local=0
            for fr in range(e.frame,stop):
                m=matches.get((fr,e.gt_id))
                if m and m[0] == e.new_tid:
                    local += 1
            support += local
            overlap_sum += e.overlap
        c.support=support
        mean_overlap=overlap_sum/max(1,len(c.edges))
        # Persistence is primary; overlap only breaks near-ties.
        c.score=float(support)+0.01*mean_overlap


def read_rows(path: Path) -> List[List[str]]:
    rows=[]
    with path.open() as f:
        for line in f:
            line=line.strip()
            if line:
                rows.append(line.split(','))
    return rows


def apply_components(rows: Sequence[Sequence[str]], selected: Sequence[Component]) -> List[List[str]]:
    by_frame: Dict[int,List[Component]]=defaultdict(list)
    for c in selected:
        by_frame[c.frame].append(c)
    # Mapping internal tracker ID -> emitted label. Always maintained as a permutation.
    mapping: Dict[int,int]={}
    out=[]
    frames=sorted({int(float(r[0])) for r in rows})
    rows_by_frame: Dict[int,List[Sequence[str]]]=defaultdict(list)
    for r in rows:
        rows_by_frame[int(float(r[0]))].append(r)
    for fr in frames:
        for c in sorted(by_frame.get(fr,[]),key=lambda x:x.index):
            tids=list(c.tids)
            for tid in tids:
                mapping.setdefault(tid,tid)
            current={tid:mapping[tid] for tid in tids}
            desired: Dict[int,int]={}
            used_labels=set()
            valid=True
            for e in c.edges:
                label=current[e.prev_tid]
                if e.new_tid in desired and desired[e.new_tid] != label:
                    valid=False; break
                if label in used_labels and desired.get(e.new_tid) != label:
                    valid=False; break
                desired[e.new_tid]=label
                used_labels.add(label)
            if not valid:
                continue
            remaining_tids=[tid for tid in tids if tid not in desired]
            remaining_labels=[current[tid] for tid in tids if current[tid] not in used_labels]
            # Preserve current labels where possible, then deterministically fill.
            assigned=set()
            for tid in remaining_tids:
                if current[tid] in remaining_labels and current[tid] not in assigned:
                    desired[tid]=current[tid]; assigned.add(current[tid])
            free_tids=[tid for tid in remaining_tids if tid not in desired]
            free_labels=[lab for lab in remaining_labels if lab not in assigned]
            for tid,lab in zip(sorted(free_tids),sorted(free_labels)):
                desired[tid]=lab
            if len(desired) != len(tids) or len(set(desired.values())) != len(tids):
                continue
            mapping.update(desired)
        emitted=[]
        for src in rows_by_frame[fr]:
            r=list(src)
            tid=int(float(r[1])); mapping.setdefault(tid,tid)
            r[1]=str(mapping[tid])
            emitted.append(r)
        ids=[int(float(r[1])) for r in emitted]
        if len(ids) != len(set(ids)):
            raise RuntimeError(f'duplicate IDs at frame {fr}')
        out.extend(emitted)
    return out


def write_rows(path: Path, rows: Iterable[Sequence[str]]) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w') as f:
        for r in rows:
            f.write(','.join(r)+'\n')


def validate_non_id(source: Sequence[Sequence[str]], repaired: Sequence[Sequence[str]]) -> dict:
    if len(source)!=len(repaired):
        raise RuntimeError('row count changed')
    mismatch=0; changed=0
    for a,b in zip(source,repaired):
        if a[0]!=b[0] or a[2:]!=b[2:]: mismatch+=1
        if a[1]!=b[1]: changed+=1
    return {'rows':len(source),'changed_id_rows':changed,'non_id_mismatch_rows':mismatch,'pass':mismatch==0}


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--seq',default='MOT20-02')
    ap.add_argument('--track-file',required=True)
    ap.add_argument('--events-csv',required=True)
    ap.add_argument('--matches-csv',required=True)
    ap.add_argument('--out-dir',required=True)
    ap.add_argument('--fractions',nargs='+',type=float,default=[0.1,0.3,0.5,1.0])
    ap.add_argument('--max-horizon',type=int,default=300)
    args=ap.parse_args()
    source_path=Path(args.track_file); out_dir=Path(args.out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    events=load_events(Path(args.events_csv),args.seq)
    matches=load_matches(Path(args.matches_csv),args.seq)
    comps=build_components(events)
    score_components(comps,events,matches,args.max_horizon)
    ranked=sorted(comps,key=lambda c:(c.score,c.support,len(c.edges),-c.frame),reverse=True)
    source=read_rows(source_path)
    manifest={'seq':args.seq,'source':str(source_path),'source_sha256':sha256(source_path),'raw_idsw_events':len(events),'conflict_components':len(comps),'max_horizon':args.max_horizon,'fractions':{},'ranking':[]}
    for rank,c in enumerate(ranked,1):
        manifest['ranking'].append({'rank':rank,'frame':c.frame,'index':c.index,'tids':list(c.tids),'edge_count':len(c.edges),'support':c.support,'score':c.score,'edges':[e.__dict__ for e in c.edges]})
    with (out_dir/'conflict_components.csv').open('w',newline='') as f:
        fields=['rank','frame','index','tids','edge_count','support','score']
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for rank,c in enumerate(ranked,1):
            w.writerow({'rank':rank,'frame':c.frame,'index':c.index,'tids':'|'.join(map(str,c.tids)),'edge_count':len(c.edges),'support':c.support,'score':c.score})
    for frac in args.fractions:
        n=max(1,round(len(ranked)*frac)) if frac>0 else 0
        selected=ranked[:n]
        repaired=apply_components(source,selected)
        tag=f'f{int(round(frac*100)):03d}'
        path=out_dir/f'{args.seq}_{tag}.txt'
        write_rows(path,repaired)
        val=validate_non_id(source,repaired)
        val.update({'fraction':frac,'selected_components':n,'selected_raw_edges':sum(len(c.edges) for c in selected),'output':str(path),'sha256':sha256(path)})
        manifest['fractions'][tag]=val
    (out_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'raw_idsw_events':len(events),'conflict_components':len(comps),'fractions':manifest['fractions'],'top10':manifest['ranking'][:10]},indent=2))

if __name__=='__main__':
    main()

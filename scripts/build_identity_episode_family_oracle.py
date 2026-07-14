from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):
            h.update(chunk)
    return h.hexdigest()


def read_track_rows(path: Path) -> List[List[str]]:
    rows=[]
    with path.open() as f:
        for line in f:
            line=line.strip()
            if line:
                rows.append(line.split(','))
    return rows


def load_matches(path: Path, seq: str) -> List[dict]:
    rows=[]
    with path.open(newline='') as f:
        for r in csv.DictReader(f):
            if r['seq']==seq:
                rows.append({
                    'frame':int(r['frame']),
                    'track_id':int(r['track_id']),
                    'gt_id':int(r['gt_id']),
                    'iou':float(r['iou']),
                })
    return rows


def build_gt_stats(matches: Sequence[dict]) -> List[dict]:
    by_gt: Dict[int,List[dict]]=defaultdict(list)
    for r in matches:
        by_gt[r['gt_id']].append(r)
    stats=[]
    for gt_id, rows in by_gt.items():
        rows=sorted(rows,key=lambda x:(x['frame'],x['track_id']))
        counts=Counter(r['track_id'] for r in rows)
        dominant_tid, dominant_rows=counts.most_common(1)[0]
        switches=0
        prev_tid=None
        prev_frame=None
        episode_count=0
        for r in rows:
            if prev_tid is None or r['track_id']!=prev_tid or (prev_frame is not None and r['frame']>prev_frame+1):
                episode_count+=1
                if prev_tid is not None and r['track_id']!=prev_tid:
                    switches+=1
            prev_tid=r['track_id']; prev_frame=r['frame']
        debt=len(rows)-dominant_rows
        stats.append({
            'gt_id':gt_id,
            'matched_rows':len(rows),
            'unique_tracker_ids':len(counts),
            'dominant_track_id':dominant_tid,
            'dominant_rows':dominant_rows,
            'non_dominant_rows':debt,
            'switches':switches,
            'episode_count':episode_count,
            'first_frame':rows[0]['frame'],
            'last_frame':rows[-1]['frame'],
            'mean_iou':sum(r['iou'] for r in rows)/len(rows),
            # Oracle ranking: association debt first, then fragmentation.
            'score':float(debt)+0.25*switches+0.01*len(counts),
        })
    stats.sort(key=lambda x:(x['score'],x['matched_rows'],x['switches']),reverse=True)
    return stats


def apply_selected(
    source: Sequence[Sequence[str]],
    matches: Sequence[dict],
    selected_gt: set[int],
    label_offset: int,
) -> Tuple[List[List[str]],dict]:
    new_id_by_key={(r['frame'],r['track_id']):label_offset+r['gt_id'] for r in matches if r['gt_id'] in selected_gt}
    out=[]; changed=0; duplicate_frames=[]
    by_frame: Dict[int,List[List[str]]]=defaultdict(list)
    for src in source:
        r=list(src)
        frame=int(float(r[0])); tid=int(float(r[1]))
        new_id=new_id_by_key.get((frame,tid))
        if new_id is not None:
            if int(float(r[1]))!=new_id:
                changed+=1
            r[1]=str(new_id)
        by_frame[frame].append(r)
    for frame in sorted(by_frame):
        rows=by_frame[frame]
        ids=[int(float(r[1])) for r in rows]
        if len(ids)!=len(set(ids)):
            duplicate_frames.append(frame)
        out.extend(rows)
    if duplicate_frames:
        raise RuntimeError(f'duplicate emitted IDs in frames: {duplicate_frames[:10]}')
    return out,{
        'changed_id_rows':changed,
        'selected_matched_rows':len(new_id_by_key),
        'duplicate_id_frames':len(duplicate_frames),
    }


def validate(source: Sequence[Sequence[str]], out: Sequence[Sequence[str]]) -> dict:
    if len(source)!=len(out):
        raise RuntimeError('row count changed')
    non_id=0
    for a,b in zip(source,out):
        if a[0]!=b[0] or list(a[2:])!=list(b[2:]):
            non_id+=1
    return {'rows':len(source),'non_id_mismatch_rows':non_id,'pass':non_id==0}


def write_rows(path: Path, rows: Iterable[Sequence[str]]) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w') as f:
        for r in rows:
            f.write(','.join(r)+'\n')


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--seq',required=True)
    ap.add_argument('--track-file',required=True)
    ap.add_argument('--matches-csv',required=True)
    ap.add_argument('--out-dir',required=True)
    ap.add_argument('--fractions',nargs='+',type=float,default=[0.1,0.3,0.5,1.0])
    ap.add_argument('--label-offset',type=int,default=2000000)
    args=ap.parse_args()

    track_path=Path(args.track_file)
    out_dir=Path(args.out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    source=read_track_rows(track_path)
    matches=load_matches(Path(args.matches_csv),args.seq)
    stats=build_gt_stats(matches)
    repairable=[r for r in stats if r['non_dominant_rows']>0 or r['switches']>0]

    with (out_dir/'identity_family_ranking.csv').open('w',newline='') as f:
        fields=list(stats[0].keys()) if stats else ['gt_id']
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(stats)

    report={
        'seq':args.seq,
        'source':str(track_path),
        'source_sha256':sha256(track_path),
        'matched_rows':len(matches),
        'gt_identities':len(stats),
        'repairable_gt_identities':len(repairable),
        'label_offset':args.label_offset,
        'ranking_unit':'GT identity family',
        'ranking_score':'non_dominant_rows + 0.25*switches + 0.01*unique_tracker_ids',
        'fractions':{},
        'top20':repairable[:20],
    }
    for frac in args.fractions:
        n=max(1,round(len(repairable)*frac)) if frac>0 else 0
        selected=repairable[:n]
        selected_gt={r['gt_id'] for r in selected}
        repaired,meta=apply_selected(source,matches,selected_gt,args.label_offset)
        tag=f'f{int(round(frac*100)):03d}'
        path=out_dir/f'{args.seq}_{tag}.txt'
        write_rows(path,repaired)
        meta.update(validate(source,repaired))
        meta.update({
            'fraction':frac,
            'selected_gt_identities':n,
            'selected_gt_ids':sorted(selected_gt),
            'selected_association_debt':sum(r['non_dominant_rows'] for r in selected),
            'selected_switches':sum(r['switches'] for r in selected),
            'output':str(path),
            'sha256':sha256(path),
        })
        report['fractions'][tag]=meta
    (out_dir/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='top20'},indent=2))
    print('top5=',json.dumps(report['top20'][:5],indent=2))

if __name__=='__main__':
    main()

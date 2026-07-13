#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from collections import defaultdict


def read_mot(path: Path):
    rows=[]
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if not line: continue
            p=line.split(',')
            if len(p)<6: continue
            fr=int(float(p[0])); tid=int(float(p[1]))
            rows.append({'frame':fr,'track_id':tid,'parts':p})
    return rows


def as_int(v, default=0):
    try: return int(float(v))
    except Exception: return default


def as_float(v, default=0.0):
    try: return float(v)
    except Exception: return default


def true_tid_from_event(row):
    rank=as_int(row.get('rank_true_before'), -1)
    if rank < 0: return -1
    tids=[x for x in str(row.get('candidate_tids','')).split('|') if x!='']
    if rank >= len(tids): return -1
    return as_int(tids[rank], -1)


def policy_match(row, policy):
    contains=as_int(row.get('contains_true_before_topk'),0)==1
    if not contains: return False
    if true_tid_from_event(row) < 0: return False
    chosen=as_int(row.get('chosen_tid'),-1)
    target=true_tid_from_event(row)
    if chosen == target: return False
    low005=as_int(row.get('low_margin_005'),0)==1
    bad=as_int(row.get('is_bad_commit_before'),0)==1
    sw=as_int(row.get('is_track_switch'),0)==1
    idsw=as_int(row.get('is_gt_idsw'),0)==1
    if policy == 'bad_contains':
        return bad
    if policy == 'bad_low005_contains':
        return bad and low005
    if policy == 'idsw_contains':
        return idsw
    if policy == 'idsw_low005_contains':
        return idsw and low005
    if policy == 'track_switch_contains':
        return sw
    if policy == 'track_switch_low005_contains':
        return sw and low005
    if policy == 'low005_wrong_contains':
        return low005 and as_int(row.get('correct_chosen_before'),0)==0
    raise ValueError(policy)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--track-file', required=True)
    ap.add_argument('--events-csv', required=True)
    ap.add_argument('--out-file', required=True)
    ap.add_argument('--policy', required=True, choices=[
        'bad_contains','bad_low005_contains','idsw_contains','idsw_low005_contains',
        'track_switch_contains','track_switch_low005_contains','low005_wrong_contains'])
    ap.add_argument('--skip-collisions', action='store_true')
    ap.add_argument('--summary-json', required=True)
    args=ap.parse_args()
    rows=read_mot(Path(args.track_file))
    # row lookup and frame occupied ids
    row_by_key={}
    occupied=defaultdict(set)
    for idx,r in enumerate(rows):
        key=(r['frame'], r['track_id'])
        if key not in row_by_key:
            row_by_key[key]=idx
        occupied[r['frame']].add(r['track_id'])
    corrections=[]
    stats=defaultdict(int)
    with open(args.events_csv, newline='', encoding='utf-8') as f:
        for ev in csv.DictReader(f):
            if not policy_match(ev,args.policy):
                continue
            fr=as_int(ev.get('frame'),-1); chosen=as_int(ev.get('chosen_tid'),-1); target=true_tid_from_event(ev)
            if fr<0 or chosen<0 or target<0 or chosen==target:
                continue
            stats['eligible_events']+=1
            key=(fr,chosen)
            if key not in row_by_key:
                stats['missing_output_row']+=1
                continue
            if args.skip_collisions and target in occupied.get(fr,set()):
                stats['skipped_collision']+=1
                continue
            corrections.append((row_by_key[key], fr, chosen, target, ev))
            # Update occupancy to avoid repeated target collisions in same frame.
            occupied[fr].discard(chosen); occupied[fr].add(target)
    # Apply corrections. Last correction for same row wins, but normally there is one.
    applied_by_idx={}
    for idx,fr,chosen,target,ev in corrections:
        applied_by_idx[idx]=(fr,chosen,target,ev)
    for idx,(fr,chosen,target,ev) in applied_by_idx.items():
        rows[idx]['parts'][1]=str(target)
    out=Path(args.out_file); out.parent.mkdir(parents=True,exist_ok=True)
    rows.sort(key=lambda r:(int(float(r['parts'][0])), int(float(r['parts'][1])), float(r['parts'][2]), float(r['parts'][3])))
    with out.open('w',encoding='utf-8') as f:
        for r in rows:
            f.write(','.join(r['parts'])+'\n')
    stats['applied_corrections']=len(applied_by_idx)
    stats['policy']=args.policy
    stats['skip_collisions']=bool(args.skip_collisions)
    # event-type stats among applied
    for _,_,_,ev in applied_by_idx.values():
        for k in ['is_bad_commit_before','is_track_switch','is_gt_idsw','low_margin_005']:
            stats['applied_'+k]+=as_int(ev.get(k),0)
    Path(args.summary_json).write_text(json.dumps(dict(stats),indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps(dict(stats),indent=2,sort_keys=True))

if __name__=='__main__': main()

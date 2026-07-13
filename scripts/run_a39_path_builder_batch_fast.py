#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, json, sys, importlib.util, traceback
from collections import defaultdict
from pathlib import Path
import numpy as np

REPO=Path(__file__).resolve().parents[1]

def load_mod(name, path):
    spec=importlib.util.spec_from_file_location(name, str(path))
    mod=importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod

pb=load_mod('pb_a39_path', REPO/'scripts'/'simulate_a39_path_bridge_rewrite.py')
scan=load_mod('scan_a39_path', REPO/'scripts'/'run_a39_path_missing_feature_scan.py')

def ai(v,d=0):
    try: return int(float(v))
    except Exception: return d

def read_csv(path: Path):
    if not path.exists(): return []
    with path.open(newline='',encoding='utf-8') as f: return list(csv.DictReader(f))

def write_any(path, rows):
    if rows:
        fields=list(rows[0].keys())
    else:
        fields=['empty']
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); w.writeheader(); w.writerows(rows)

def load_jobs(stage_report: Path):
    rows=read_csv(stage_report)
    jobs=[r for r in rows if r.get('stage')=='feature_missing_cached' and r.get('candidate_family')=='path_index']
    def pri(r):
        return (1 if r.get('index_mode')=='bridge_or_direct_index' else 0, scan.af(r.get('sim')), -ai(r.get('row_rank'),999), -ai(r.get('col_rank'),999))
    jobs=sorted(jobs,key=pri,reverse=True)
    for i,r in enumerate(jobs,1): r['job_rank']=i
    return jobs

def build_one(job, run_dir, args, globals_):
    rows=globals_['rows']; rows_by=globals_['rows_by']; by_frame_tid=globals_['by_frame_tid']; by_tid=globals_['by_tid']; row_gt=globals_['row_gt']; encoder=globals_['encoder']; tunnels=globals_['tunnels']
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir/'rewrite_summary.json').exists():
        return 'reused_existing'
    tunnel=tunnels[ai(job.get('tunnel_id'))]
    start,end=tunnel['start'],tunnel['end']; f0,f1=start,end+args.exit_window
    pre=ai(job.get('pre_track')); post=ai(job.get('post_track')); target_id=pre
    pre_anchor_rows=[r for r in by_tid.get(pre,[]) if start-args.pre_window <= r['frame'] < start]
    post_anchor_rows=[r for r in by_tid.get(post,[]) if end < r['frame'] <= end+args.post_window]
    if not pre_anchor_rows or not post_anchor_rows:
        raise RuntimeError('missing pre or post anchor clean rows')
    groups={'anchor_pre':pre_anchor_rows,'anchor_post':post_anchor_rows}
    fragments=[]
    for tid in sorted(tunnel['tracks']):
        frag_rows_all=[r for r in by_tid.get(tid,[]) if f0 <= r['frame'] <= f1]
        for local_id, frag_rows in enumerate(pb.split_fragments(frag_rows_all, max_gap=1)):
            if not frag_rows: continue
            key=f'frag_{tid}_{local_id}'
            fragments.append((key,tid,local_id,frag_rows)); groups[key]=frag_rows
    proto,crop_counts=pb.extract_features(groups,args.img_dir,encoder,args.max_crops_per_fragment)
    if 'anchor_pre' not in proto or 'anchor_post' not in proto:
        raise RuntimeError('missing anchor features')
    anchor_proto=proto['anchor_pre']+proto['anchor_post']; anchor_proto=anchor_proto/max(float(np.linalg.norm(anchor_proto)),1e-12)
    metas=[]; meta_by_key={}
    for key,tid,local_id,rr in fragments:
        if key not in proto: continue
        m=pb.frag_meta(key,tid,local_id,rr,row_gt,crop_counts)
        m['target_id']=target_id
        m['sim_to_pre']=float(np.dot(proto[key],proto['anchor_pre']))
        m['sim_to_post']=float(np.dot(proto[key],proto['anchor_post']))
        m['sim_to_anchor']=float(np.dot(proto[key],anchor_proto))
        m['collision_rows_if_rewrite']=sum(1 for r in rr if tid != target_id and by_frame_tid.get((r['frame'],target_id)))
        m['collision_ratio_if_rewrite']=pb.safe_div(m['collision_rows_if_rewrite'],len(rr))
        m['gt_same_as_anchor']=0
        m['selected_stage']=''; m['bridge_score']=''
        metas.append(m); meta_by_key[key]=m
    selected=[]
    for m in metas:
        if m['track_id']==target_id: continue
        if m['collision_ratio_if_rewrite']>0: continue
        if max(m['sim_to_anchor'],m['sim_to_post']) >= args.high_sim and m['rows']>=2:
            m['selected_stage']='high_reid'; selected.append(m)
    selected=sorted(selected,key=lambda x:(x['frame_start'],x['frame_end']))
    target_frags=[m for m in metas if m['track_id']==target_id and m['frame_start'] <= (selected[0]['frame_start'] if selected else f1)]
    if target_frags: prev_ep=pb.endpoint_from_meta(sorted(target_frags,key=lambda x:x['frame_end'])[-1],'last')
    else: prev_ep=pb.endpoint_from_rows(pre_anchor_rows,'last')
    bridge_candidates=[]
    if selected:
        next_m=selected[0]; next_ep=pb.endpoint_from_meta(next_m,'first')
        for m in metas:
            if m['track_id']==target_id or m in selected: continue
            if m['collision_ratio_if_rewrite']>0: continue
            if not (prev_ep['frame'] < m['frame_start'] <= m['frame_end'] < next_ep['frame']): continue
            if m['frame_start']-prev_ep['frame'] > args.bridge_max_gap: continue
            if next_ep['frame']-m['frame_end'] > args.bridge_max_gap: continue
            score,d1,d2,h1,h2=pb.bridge_score(prev_ep,next_ep,m)
            mrow={'fragment_key':m['fragment_key'],'track_id':m['track_id'],'frame_start':m['frame_start'],'frame_end':m['frame_end'],'rows':m['rows'],'sim_to_anchor':m['sim_to_anchor'],'sim_to_pre':m['sim_to_pre'],'sim_to_post':m['sim_to_post'],'bridge_score':score,'start_dist':d1,'end_dist':d2,'height_ratio_start':h1,'height_ratio_end':h2,'accepted':0,'major_gt':m['major_gt'],'gt_purity':m['gt_purity'],'gt_same_as_anchor':m['gt_same_as_anchor']}
            if m['sim_to_anchor'] >= args.bridge_min_sim and score <= args.bridge_max_score: mrow['accepted']=1
            bridge_candidates.append(mrow)
        accepted_bridge=[r for r in bridge_candidates if r['accepted']]
        if accepted_bridge:
            best=sorted(accepted_bridge,key=lambda r:(r['bridge_score'],-r['frame_end'],-r['rows']))[0]
            bm=meta_by_key[best['fragment_key']]; bm['selected_stage']='bridge_fragment'; bm['bridge_score']=best['bridge_score']
            selected=sorted([bm]+selected,key=lambda x:(x['frame_start'],x['frame_end']))
    gap_rows=[]; selected_for_gaps=sorted(selected,key=lambda x:(x['frame_start'],x['frame_end']))
    for a,b in zip(selected_for_gaps,selected_for_gaps[1:]):
        a_ep=pb.endpoint_from_meta(a,'last'); b_ep=pb.endpoint_from_meta(b,'first')
        if b_ep['frame']-a_ep['frame'] <= 1: continue
        for fr in range(a_ep['frame']+1,b_ep['frame']):
            pred=pb.predict_between(a_ep,b_ep,fr); cand=[]
            for r in rows_by.get(fr,[]):
                if r['track_id'] not in tunnel['tracks'] or r['track_id']==target_id: continue
                if by_frame_tid.get((fr,target_id)): continue
                dist=pb.norm_dist(r,pred); height_ratio=r['height']/max(pred['height'],1e-6)
                if dist <= args.gap_max_dist and args.gap_height_min <= height_ratio <= args.gap_height_max:
                    cand.append((dist,abs(1.0-height_ratio),r,height_ratio))
            if cand:
                dist,hpen,r,height_ratio=sorted(cand,key=lambda x:(x[0],x[1],-x[2]['score']))[0]
                gap_rows.append({'frame':fr,'track_id':r['track_id'],'idx':r['idx'],'target_id':target_id,'dist':dist,'height_ratio':height_ratio,'score':r['score'],'gt_id':row_gt.get(r['idx'],-1),'gt_same_as_anchor':0,'bridge_from':a['fragment_key'],'bridge_to':b['fragment_key']})
    selected_indices=[]
    for m in selected_for_gaps:
        if m['track_id'] != target_id: selected_indices.extend(m['row_indices'])
    selected_indices.extend([r['idx'] for r in gap_rows]); selected_indices=sorted(set(selected_indices),key=lambda idx:rows[idx]['frame'])
    final_id={r['idx']:r['track_id'] for r in rows}; cur=defaultdict(set)
    for r in rows: cur[r['frame']].add(r['track_id'])
    applied=0; skipped=0
    for idx in selected_indices:
        r=rows[idx]
        if r['track_id']==target_id: continue
        if target_id in cur[r['frame']]: skipped+=1; continue
        cur[r['frame']].discard(r['track_id']); cur[r['frame']].add(target_id); final_id[idx]=target_id; applied+=1
    selected_frag_rows=[]
    for m in sorted(selected_for_gaps,key=lambda x:(x['frame_start'],x['frame_end'])):
        selected_frag_rows.append({k:v for k,v in m.items() if k!='row_indices'})
    all_frag_rows=[{k:v for k,v in m.items() if k!='row_indices'} for m in metas]
    write_any(run_dir/'all_fragments.csv',all_frag_rows); write_any(run_dir/'selected_fragments.csv',selected_frag_rows); write_any(run_dir/'bridge_candidates.csv',bridge_candidates); write_any(run_dir/'gap_rows.csv',gap_rows)
    track_out=run_dir/'track_results'/'MOT20-02.txt'; track_out.parent.mkdir(parents=True,exist_ok=True)
    with track_out.open('w',encoding='utf-8') as f:
        out_rows=[]
        for r in rows:
            p=list(r['parts']); p[1]=str(final_id[r['idx']]); out_rows.append(p)
        for p in sorted(out_rows,key=lambda p:(ai(p[0]),ai(p[1]),float(p[2]),float(p[3]))): f.write(','.join(p)+'\n')
    summary={'tunnel_id':ai(job.get('tunnel_id')),'pre_anchor':pre,'post_anchor':post,'target_id':target_id,'anchor_gt_diag':None,'selected_fragments':[m['fragment_key'] for m in selected_for_gaps],'selected_fragment_count':len(selected_for_gaps),'correct_fragment_count_diag':0,'wrong_fragment_count_diag':0,'gap_row_count':len(gap_rows),'correct_gap_rows_diag':0,'wrong_gap_rows_diag':0,'planned_rows':len(selected_indices),'applied_rows':applied,'skipped_collision_rows':skipped,'high_sim':args.high_sim,'bridge_min_sim':args.bridge_min_sim,'bridge_max_score':args.bridge_max_score,'gap_max_dist':args.gap_max_dist,'gt_diag_selected_rows_same_anchor':0,'gt_diag_selected_rows_wrong_or_unknown':0}
    (run_dir/'rewrite_summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    (run_dir/'rewrite_summary.md').write_text('# A39 path builder batch fast\n\n```json\n'+json.dumps(summary,indent=2,sort_keys=True)+'\n```\n')
    return 'ok'

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--stage-report',default='outputs/spot_runtime_gate_20260628/A39_identity_suspension_tunnel/A39_06d_full_seq_no_gt_candidate_scan_seq02/transaction_stage_report_full_nogt_cached.csv')
    ap.add_argument('--out-dir',default='outputs/spot_runtime_gate_20260628/A39_identity_suspension_tunnel/A39_06d_full_seq_no_gt_candidate_scan_seq02/A39_06d2c_path_missing_feature_scan')
    ap.add_argument('--track-file',default='outputs/spot_runtime_gate_20260628/A37_identity_escrow_association/A37_00b_candidate_logger_seq02/track_results/MOT20-02.txt')
    ap.add_argument('--gt-file',default='datasets/MOT20/train/MOT20-02/gt/gt.txt')
    ap.add_argument('--tunnels-csv',default='outputs/spot_runtime_gate_20260628/A39_identity_suspension_tunnel/A39_00_tunnel_discovery_seq02_ioa050_dur3/tunnel_candidates.csv')
    ap.add_argument('--img-dir',default='datasets/MOT20/train/MOT20-02/img1')
    ap.add_argument('--fast-reid-config',default='external/BoT-SORT-main/fast_reid/configs/MOT20/sbs_S50.yml')
    ap.add_argument('--fast-reid-weights',default='external/BoT-SORT-main/pretrained/mot20_sbs_S50.pth')
    ap.add_argument('--top-k',type=int,default=20)
    ap.add_argument('--device',default='cuda')
    ap.add_argument('--pre-window',type=int,default=10); ap.add_argument('--post-window',type=int,default=10); ap.add_argument('--exit-window',type=int,default=10)
    ap.add_argument('--max-crops-per-fragment',type=int,default=8); ap.add_argument('--high-sim',type=float,default=0.65); ap.add_argument('--bridge-min-sim',type=float,default=0.35); ap.add_argument('--bridge-max-score',type=float,default=0.22); ap.add_argument('--bridge-max-gap',type=int,default=25); ap.add_argument('--gap-max-dist',type=float,default=0.12); ap.add_argument('--gap-height-min',type=float,default=0.80); ap.add_argument('--gap-height-max',type=float,default=1.25); ap.add_argument('--iou-thr',type=float,default=0.5)
    args=ap.parse_args(); out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    jobs=load_jobs(Path(args.stage_report)); completed={p.parent.name for p in (out/'cases').glob('*/rewrite_summary.json')} | {p.parent.name for p in (out/'cases').glob('*/feature_failed.json')}
    jobs=[j for j in jobs if j['transaction_id'] not in completed][:args.top_k]
    rows,rows_by,by_frame_tid,by_tid,img_width=pb.read_track(Path(args.track_file)); gt_by=pb.read_gt(Path(args.gt_file)); row_gt,row_iou=pb.match_rows(rows_by,gt_by,args.iou_thr)
    tunnels={ai(r.get('tunnel_id')):{'tunnel_id':ai(r.get('tunnel_id')),'start':ai(r.get('start')),'end':ai(r.get('end')),'duration':ai(r.get('duration')),'tracks':set(ai(x) for x in str(r.get('tracks','')).split('|') if x!='')} for r in read_csv(Path(args.tunnels_csv))}
    encoder=pb.FastReIDInterface(args.fast_reid_config,args.fast_reid_weights,args.device,batch_size=32)
    g={'rows':rows,'rows_by':rows_by,'by_frame_tid':by_frame_tid,'by_tid':by_tid,'row_gt':row_gt,'encoder':encoder,'tunnels':tunnels}
    scanned=[]
    for i,job in enumerate(jobs,1):
        tid=job['transaction_id']; rd=out/'cases'/tid
        try:
            status=build_one(job,rd,args,g)
        except Exception as e:
            status='path_builder_failed_fast'; rd.mkdir(parents=True,exist_ok=True); (rd/'feature_failed.json').write_text(json.dumps({'transaction_id':tid,'error':repr(e),'traceback':traceback.format_exc()},indent=2)+'\n')
        rec=scan.summarize(job,rd,status); (rd/'feature_summary_nogt.json').write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n')
        scanned.append(rec); print(f'[{i}/{len(jobs)}] {tid} {status} mode={rec.get("mode")} planned={rec.get("planned_rows")}')
    print(json.dumps({'scanned':len(scanned),'ids':[r.get('transaction_id') for r in scanned]},indent=2))

if __name__=='__main__': main()

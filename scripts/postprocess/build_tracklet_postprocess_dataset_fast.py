#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, math, statistics
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np


def mean(xs): return float(sum(xs)/len(xs)) if xs else 0.0
def std(xs): return float(statistics.pstdev(xs)) if len(xs)>1 else 0.0
def median(xs): return float(statistics.median(xs)) if xs else 0.0

def load_mot_by_track_and_frame(path: Path):
    tracks=defaultdict(list); by_frame=defaultdict(list)
    with path.open() as f:
        for line in f:
            line=line.strip()
            if not line: continue
            p=line.split(',')
            if len(p)<6: continue
            try:
                r={'frame':int(float(p[0])),'tid':int(float(p[1])),'x':float(p[2]),'y':float(p[3]),'w':float(p[4]),'h':float(p[5]),'score':float(p[6]) if len(p)>6 else 1.0}
            except Exception:
                continue
            tracks[r['tid']].append(r); by_frame[r['frame']].append(r)
    for t in tracks.values(): t.sort(key=lambda r:r['frame'])
    return tracks, by_frame

def load_gt_by_frame(path: Path):
    by=defaultdict(list)
    with path.open() as f:
        for line in f:
            line=line.strip()
            if not line: continue
            p=line.split(',')
            if len(p)<6: continue
            try:
                fr=int(float(p[0])); gid=int(float(p[1])); x,y,w,h=map(float,p[2:6]); mark=int(float(p[6])) if len(p)>6 else 1; cls=int(float(p[7])) if len(p)>7 else 1
            except Exception:
                continue
            if mark==1 and cls==1: by[fr].append((gid,x,y,w,h))
    return by

def annotate_by_iou(pred_by_frame, gt_by_frame, iou_thr):
    for fr, rows in pred_by_frame.items():
        gts=gt_by_frame.get(fr, [])
        if not gts:
            for r in rows: r['best_gt']=-1; r['best_iou']=0.0; r['matched']=0
            continue
        p=np.array([[r['x'],r['y'],r['w'],r['h']] for r in rows], dtype=float)
        g=np.array([[x,y,w,h] for _,x,y,w,h in gts], dtype=float)
        gids=np.array([gid for gid,_,_,_,_ in gts], dtype=int)
        px1=p[:,0][:,None]; py1=p[:,1][:,None]; px2=(p[:,0]+p[:,2])[:,None]; py2=(p[:,1]+p[:,3])[:,None]
        gx1=g[:,0][None,:]; gy1=g[:,1][None,:]; gx2=(g[:,0]+g[:,2])[None,:]; gy2=(g[:,1]+g[:,3])[None,:]
        iw=np.maximum(0.0, np.minimum(px2,gx2)-np.maximum(px1,gx1))
        ih=np.maximum(0.0, np.minimum(py2,gy2)-np.maximum(py1,gy1))
        inter=iw*ih
        pa=(p[:,2]*p[:,3])[:,None]; ga=(g[:,2]*g[:,3])[None,:]
        union=np.maximum(1e-9, pa+ga-inter)
        ious=inter/union
        idx=ious.argmax(axis=1); vals=ious[np.arange(len(rows)), idx]
        for r, biou, j in zip(rows, vals, idx):
            r['best_iou']=float(biou); r['best_gt']=int(gids[j]) if biou>=iou_thr else -1; r['matched']=int(biou>=iou_thr)

def center(r): return (r['x']+r['w']/2.0, r['y']+r['h']/2.0)
def area(r): return max(0.0,r['w'])*max(0.0,r['h'])
def bottom(r): return r['y']+r['h']
def endpoint_velocity(rows, side, k=5):
    if len(rows)<2: return 0.0,0.0,0.0
    sub=rows[-k:] if side=='end' else rows[:k]
    if len(sub)<2: sub=rows[-2:] if side=='end' else rows[:2]
    a,b=sub[0],sub[-1]; dt=max(1,b['frame']-a['frame']); ca,cb=center(a),center(b)
    vx=(cb[0]-ca[0])/dt; vy=(cb[1]-ca[1])/dt
    return vx,vy,math.hypot(vx,vy)

def track_stats(seq, condition, tid, rows, original_keys=None):
    n=len(rows); frames=[r['frame'] for r in rows]; scores=[r['score'] for r in rows]; areas=[area(r) for r in rows]; hs=[r['h'] for r in rows]; ws=[r['w'] for r in rows]; bottoms=[bottom(r) for r in rows]
    interp_count=sum(0 if original_keys is None or (tid,r['frame']) in original_keys else 1 for r in rows)
    gaps=[frames[i+1]-frames[i] for i in range(len(frames)-1)]
    speeds=[]
    for a,b in zip(rows,rows[1:]):
        dt=max(1,b['frame']-a['frame']); ca,cb=center(a),center(b); speeds.append(math.hypot(cb[0]-ca[0], cb[1]-ca[1])/dt)
    matched=sum(r.get('matched',0) for r in rows); gts=[r.get('best_gt',-1) for r in rows if r.get('best_gt',-1)>=0]
    c=Counter(gts); dom,domc=(-1,0) if not c else c.most_common(1)[0]
    gt_switches=0; prev=None
    for g in gts:
        if prev is not None and g!=prev: gt_switches+=1
        prev=g
    mr=matched/n if n else 0.0; drt=domc/n if n else 0.0; drm=domc/matched if matched else 0.0
    label='ambiguous_track'
    if n>=2 and mr>=0.6 and drt>=0.6: label='good_track'
    elif mr<=0.2: label='bad_track'
    vxs,vys,ss=endpoint_velocity(rows,'start'); vxe,vye,se=endpoint_velocity(rows,'end')
    return {'condition':condition,'seq':seq,'track_id':tid,'row_count':n,'start_frame':min(frames),'end_frame':max(frames),'duration':max(frames)-min(frames)+1,'num_gaps':sum(1 for g in gaps if g>1),'missing_gap_frames':sum(max(0,g-1) for g in gaps),'max_gap':max([0]+[g-1 for g in gaps]),'interpolated_count':interp_count,'interpolated_fraction':interp_count/n if n else 0.0,'avg_score':mean(scores),'min_score':min(scores),'max_score':max(scores),'median_score':median(scores),'score_std':std(scores),'avg_area':mean(areas),'median_area':median(areas),'area_std':std(areas),'avg_height':mean(hs),'median_height':median(hs),'height_std':std(hs),'avg_width':mean(ws),'avg_bottom_y':mean(bottoms),'avg_center_speed':mean(speeds),'max_center_speed':max(speeds) if speeds else 0.0,'center_speed_std':std(speeds),'vx_start':vxs,'vy_start':vys,'speed_start':ss,'vx_end':vxe,'vy_end':vye,'speed_end':se,'matched_count':matched,'matched_ratio':mr,'dominant_gt':dom,'dominant_gt_count':domc,'dominant_gt_ratio_total':drt,'dominant_gt_ratio_matched':drm,'gt_switches_inside_track':gt_switches,'quality_label':label,'first_x':rows[0]['x'],'first_y':rows[0]['y'],'first_w':rows[0]['w'],'first_h':rows[0]['h'],'first_score':rows[0]['score'],'last_x':rows[-1]['x'],'last_y':rows[-1]['y'],'last_w':rows[-1]['w'],'last_h':rows[-1]['h'],'last_score':rows[-1]['score']}

def pair_features(a,b):
    gap=b['start_frame']-a['end_frame']; ax=a['last_x']+a['last_w']/2; ay=a['last_y']+a['last_h']/2; bx=b['first_x']+b['first_w']/2; by=b['first_y']+b['first_h']/2
    dist=math.hypot(bx-ax,by-ay); predx=ax+a['vx_end']*gap; predy=ay+a['vy_end']*gap; pd=math.hypot(bx-predx,by-predy)
    na=math.hypot(a['vx_end'],a['vy_end']); nb=math.hypot(b['vx_start'],b['vy_start']); vc=(a['vx_end']*b['vx_start']+a['vy_end']*b['vy_start'])/(na*nb) if na>1e-6 and nb>1e-6 else 0.0
    ar_a=max(1e-6,a['last_w']*a['last_h']); ar_b=max(1e-6,b['first_w']*b['first_h'])
    return {'seq':a['seq'],'track_a':a['track_id'],'track_b':b['track_id'],'gap':gap,'center_distance':dist,'center_distance_per_frame':dist/max(1,gap),'predicted_distance':pd,'predicted_distance_per_frame':pd/max(1,gap),'velocity_cosine':vc,'height_ratio':max(a['last_h'],b['first_h'])/max(1e-6,min(a['last_h'],b['first_h'])),'area_ratio':max(ar_a,ar_b)/max(1e-6,min(ar_a,ar_b)),'bottom_y_gap':abs((a['last_y']+a['last_h'])-(b['first_y']+b['first_h'])),'len_a':a['row_count'],'len_b':b['row_count'],'duration_a':a['duration'],'duration_b':b['duration'],'avg_score_a':a['avg_score'],'avg_score_b':b['avg_score'],'last_score_a':a['last_score'],'first_score_b':b['first_score'],'matched_ratio_a':a['matched_ratio'],'matched_ratio_b':b['matched_ratio'],'dominant_gt_a':a['dominant_gt'],'dominant_gt_b':b['dominant_gt'],'same_gt':int(a['dominant_gt']>=0 and a['dominant_gt']==b['dominant_gt']),'quality_label_a':a['quality_label'],'quality_label_b':b['quality_label']}

def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows: path.write_text(''); return
    with path.open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--online-dir',required=True); ap.add_argument('--interp-dir',required=True); ap.add_argument('--gt-root',required=True); ap.add_argument('--out-dir',required=True); ap.add_argument('--iou-thresh',type=float,default=0.5); ap.add_argument('--max-link-gap',type=int,default=60); ap.add_argument('--max-center-step',type=float,default=80); ap.add_argument('--max-area-ratio',type=float,default=4); ap.add_argument('--min-track-len-for-link',type=int,default=5); args=ap.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    quality=[]; tracklets=[]; pairs=[]
    seqs=sorted(p.stem for p in Path(args.online_dir).glob('MOT20-*.txt'))
    for seq in seqs:
        gt=load_gt_by_frame(Path(args.gt_root)/seq/'gt'/'gt.txt')
        on_tracks,on_by=load_mot_by_track_and_frame(Path(args.online_dir)/f'{seq}.txt'); annotate_by_iou(on_by,gt,args.iou_thresh)
        in_tracks,in_by=load_mot_by_track_and_frame(Path(args.interp_dir)/f'{seq}.txt'); annotate_by_iou(in_by,gt,args.iou_thresh)
        on_keys={(tid,r['frame']) for tid,rs in on_tracks.items() for r in rs}
        seq_stats=[]
        for tid,rs in sorted(on_tracks.items()):
            st=track_stats(seq,'online',tid,rs); quality.append(st); tracklets.append(st); seq_stats.append(st)
        for tid,rs in sorted(in_tracks.items()): quality.append(track_stats(seq,'interp_gap30',tid,rs,on_keys))
        valid=[r for r in seq_stats if r['row_count']>=args.min_track_len_for_link and r['matched_ratio']>0]
        starts=sorted(valid, key=lambda r:r['start_frame'])
        for a in valid:
            for b in starts:
                gap=b['start_frame']-a['end_frame']
                if gap<=0: continue
                if gap>args.max_link_gap and b['start_frame']>a['end_frame']+args.max_link_gap: break
                pf=pair_features(a,b)
                if pf['center_distance_per_frame']<=args.max_center_step and pf['area_ratio']<=args.max_area_ratio: pairs.append(pf)
    write_csv(out/'track_quality_rows.csv',quality); write_csv(out/'tracklet_rows.csv',tracklets); write_csv(out/'aflink_pair_candidates.csv',pairs)
    qsum={}
    for cond in sorted(set(r['condition'] for r in quality)):
        rs=[r for r in quality if r['condition']==cond]; c=Counter(r['quality_label'] for r in rs)
        qsum[cond]={'tracks':len(rs),'good':c.get('good_track',0),'bad':c.get('bad_track',0),'ambiguous':c.get('ambiguous_track',0),'mean_matched_ratio':mean([r['matched_ratio'] for r in rs]),'mean_interpolated_fraction':mean([r['interpolated_fraction'] for r in rs]),'short_tracks_len_le_3':sum(r['row_count']<=3 for r in rs),'low_match_tracks':sum(r['matched_ratio']<=0.2 for r in rs)}
    psum={'candidate_pairs':len(pairs),'positive_pairs':sum(p['same_gt'] for p in pairs),'negative_pairs':len(pairs)-sum(p['same_gt'] for p in pairs)}; psum['positive_rate']=psum['positive_pairs']/psum['candidate_pairs'] if psum['candidate_pairs'] else 0.0; psum['by_seq']={}
    for seq in seqs:
        rs=[p for p in pairs if p['seq']==seq]; pos=sum(p['same_gt'] for p in rs); psum['by_seq'][seq]={'pairs':len(rs),'positive':pos,'negative':len(rs)-pos,'positive_rate':pos/len(rs) if rs else 0.0}
    (out/'track_quality_summary.json').write_text(json.dumps(qsum,indent=2,sort_keys=True)+'\n'); (out/'aflink_candidate_summary.json').write_text(json.dumps(psum,indent=2,sort_keys=True)+'\n')
    md=['# A22_00 Track Quality Summary','']
    for cond,s in qsum.items():
        md += [f'## {cond}','| metric | value |','|---|---:|'] + [f'| {k} | {v} |' for k,v in s.items()] + ['']
    (out/'track_quality_summary.md').write_text('\n'.join(md)+'\n')
    md=['# A22_00 AFLink Candidate Summary','','| seq | pairs | positive | negative | positive_rate |','|---|---:|---:|---:|---:|']
    for seq,s in psum['by_seq'].items(): md.append(f"| {seq} | {s['pairs']} | {s['positive']} | {s['negative']} | {s['positive_rate']:.4f} |")
    md += ['', f"total_pairs: {psum['candidate_pairs']}", f"positive_pairs: {psum['positive_pairs']}", f"positive_rate: {psum['positive_rate']:.4f}"]
    (out/'aflink_candidate_summary.md').write_text('\n'.join(md)+'\n')
    print(json.dumps({'quality_summary':qsum,'pair_summary':psum},indent=2,sort_keys=True))
if __name__=='__main__': main()

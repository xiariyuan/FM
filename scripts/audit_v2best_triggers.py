"""Comprehensive audit of DMM v2 triggers on MOT20-02."""
from __future__ import annotations
import argparse, csv, json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

def iou_xyxy(a, b):
    ix1=max(a[0],b[0]); iy1=max(a[1],b[1]); ix2=min(a[2],b[2]); iy2=min(a[3],b[3])
    iw=max(0,ix2-ix1); ih=max(0,iy2-iy1); inter=iw*ih
    ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return float(inter/ua) if ua>0 else 0.0

def load_dump(path):
    d=np.load(path, allow_pickle=True)
    dets=d["detections"]; feats=d["features"].astype(np.float32)
    cols=list(d["columns"]); gi=cols.index("global_det_idx"); fi=cols.index("frame")
    bi=cols.index("x1"); si=cols.index("score")
    by_global={}; by_frame=defaultdict(list)
    for ri,row in enumerate(dets):
        gidx=int(row[gi]); frame=int(row[fi])
        box=[float(row[bi]),float(row[bi+1]),float(row[bi+2]),float(row[bi+3])]
        by_global[gidx]=(frame,box,feats[ri],float(row[si]))
        by_frame[frame].append((gidx,box,feats[ri],float(row[si])))
    return by_global,by_frame

def load_gt(path):
    gtf=defaultdict(list)
    for line in open(path):
        p=line.strip().split(",")
        if len(p)<6: continue
        gtf[int(p[0])].append((int(p[1]),(float(p[2]),float(p[3]),float(p[2])+float(p[4]),float(p[3])+float(p[5])),float(p[8]) if len(p)>=9 else 1.0))
    return gtf

def load_tracks(path):
    t=defaultdict(list)
    for line in open(path):
        p=line.strip().split(",")
        if len(p)<6: continue
        t[int(p[1])].append((int(p[0]),(float(p[2]),float(p[3]),float(p[2])+float(p[4]),float(p[3])+float(p[5]))))
    for k in t: t[k].sort()
    return t

def match_gt(box,gts,thr=0.5):
    bid=-1; bi=0.0
    for gid,gbox,vis in gts:
        i=iou_xyxy(box,gbox)
        if i>bi: bi=i; bid=gid
    return (bid,bi) if bi>=thr else (-1,bi)

def canon_gt(rows,gtf,upto,thr=0.5):
    c=Counter()
    for f,box in rows:
        if f>=upto: continue
        gid,_=match_gt(box,gtf.get(f,[]),thr)
        if gid>0: c[gid]+=1
    return c.most_common(1)[0] if c else (-1,0)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dump",default="outputs/dmm_phase0_mot20_02/MOT20-02/dump_yolox_reid.npz")
    ap.add_argument("--gt",default="/gemini/code/datasets/MOT20/train/MOT20-02/gt/gt.txt")
    ap.add_argument("--base",default="outputs/dmm_phase1_base_mot20_02_reid/track_results/MOT20-02.txt")
    ap.add_argument("--v2",default="outputs/dmm_phase2_v2best_mot20_02/track_results/MOT20-02.txt")
    ap.add_argument("--events",default="outputs/dmm_phase2_v2best_mot20_02/dmm_events.csv")
    ap.add_argument("--out-dir",default="outputs/dmm_phase2_audit_v2best_mot20_02")
    ap.add_argument("--iou-thresh",type=float,default=0.5)
    args=ap.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)

    print("loading dump..."); dbg,dfr=load_dump(args.dump); print(f"  {len(dbg)} dets")
    print("loading GT..."); gtf=load_gt(args.gt); print(f"  {len(gtf)} frames")
    print("loading base..."); bt=load_tracks(args.base); print(f"  {len(bt)} tracks")
    print("loading v2..."); vt=load_tracks(args.v2); print(f"  {len(vt)} tracks")
    print("loading events...")
    events=list(csv.DictReader(open(args.events)))
    defers=[e for e in events if e["event"]=="defer"]
    recovers=[e for e in events if e["event"] in ("v2_recover","recover")]
    print(f"  defers={len(defers)} recovers={len(recovers)}")

    rby={int(r["track_id"]):r for r in recovers}

    # Map track output -> det_global_idx at each frame
    def map_dets(tracks):
        m=defaultdict(dict)
        for tid,rows in tracks.items():
            for f,box in rows:
                best=-1; bi=0.0
                for gidx,dbox,_,_ in dfr.get(f,[]):
                    i=iou_xyxy(box,dbox)
                    if i>bi: bi=i; best=gidx
                if bi>0.5: m[tid][f]=best
        return m
    print("mapping base dets..."); bm=map_dets(bt)
    print("mapping v2 dets..."); vm=map_dets(vt)

    rows=[]
    counts=Counter()
    for ev in defers:
        ft=int(ev["frame"]); tid=int(ev["track_id"]); dg=int(ev["det_global_idx"])
        cost=float(ev.get("cost","nan")); rm=float(ev.get("row_margin","nan")); cm=float(ev.get("col_margin","nan"))
        cs=int(ev.get("crowd_cluster_size",1)); cpc=int(ev.get("crowd_peer_count",0))
        miop=float(ev.get("max_iou_with_peer",0.0)); moap=float(ev.get("max_ioa_with_peer",0.0))
        rd=float(ev.get("reid_dist",-1.0)); ds=float(ev.get("det_score",0.0)); ta=int(ev.get("track_age",0))
        fd=len(dfr.get(ft,[]))

        if dg not in dbg:
            dgid=-1; dgi=0.0
        else:
            _,dbox,_,_=dbg[dg]
            dgid,dgi=match_gt(dbox,gtf.get(ft,[]),args.iou_thresh)

        bcg,bcn=canon_gt(bt.get(tid,[]),gtf,ft,args.iou_thresh)
        vcg,vcn=canon_gt(vt.get(tid,[]),gtf,ft,args.iou_thresh)

        bgidx=bm.get(tid,{}).get(ft,-1)
        bgt=-1; bgi=0.0
        if bgidx>=0 and bgidx in dbg:
            _,bbox,_,_=dbg[bgidx]
            bgt,bgi=match_gt(bbox,gtf.get(ft,[]),args.iou_thresh)

        vgidx=vm.get(tid,{}).get(ft,-1)
        vgt=-1; vgi=0.0
        vp=int(any(f==ft for f,_ in vt.get(tid,[])))
        if vgidx>=0 and vgidx in dbg:
            _,vbox,_,_=dbg[vgidx]
            vgt,vgi=match_gt(vbox,gtf.get(ft,[]),args.iou_thresh)

        rev=rby.get(tid)
        vc=float(rev["cost"]) if rev else -1.0
        va=float(rev.get("app_cost",-1.0)) if rev else -1.0
        vmot=float(rev.get("motion_cost",-1.0)) if rev else -1.0
        vdir=float(rev.get("direction_cost",-1.0)) if rev else -1.0
        vr=int(rev is not None)
        vg=cost-vc if vr else 0.0

        bc=int(bgt==bcg and bcg>0)
        vc2=int(vgt==vcg and vcg>0)
        bh=int(bgidx>=0)
        vh=int(vp and vgidx>=0)

        if dgid<0: v="uncertain"
        elif not vh and vp==0: v="harmful_fn"
        elif bc and vc2: v="neutral"
        elif not bc and vc2: v="beneficial_trigger"
        elif bc and not vc2: v="harmful_override"
        elif not bc and not vc2: v="neutral_both_wrong"
        else: v="uncertain"
        counts[v]+=1

        rows.append({"frame":ft,"track_id":tid,"det_global_idx":dg,"primary_cost":cost,
            "v2_cost":vc,"v2_gain":round(vg,4),"row_margin":round(rm,4),"col_margin":round(cm,4),
            "cluster_size":cs,"crowd_peer_count":cpc,"frame_density":fd,
            "max_iou_peer":round(miop,4),"max_ioa_peer":round(moap,4),"reid_dist":round(rd,4),
            "det_score":round(ds,4),"track_age":ta,"det_gt_id":dgid,"det_gt_iou":round(dgi,3),
            "base_canon_gt":bcg,"base_canon_n":bcn,"v2_canon_gt":vcg,"v2_canon_n":vcn,
            "base_present":bh,"base_gt_at_t":bgt,"base_gt_iou":round(bgi,3),"base_correct":bc,
            "v2_present":vp,"v2_gt_at_t":vgt,"v2_gt_iou":round(vgi,3),"v2_correct":vc2,
            "v2_recovered":vr,"v2_app":round(va,4),"v2_motion":round(vmot,4),"v2_dir":round(vdir,4),
            "verdict":v})

    csv_p=out/"audit_triggers.csv"
    fn=list(rows[0].keys()) if rows else ["frame"]
    with open(csv_p,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fn); w.writeheader(); w.writerows(rows)
    print(f"\naudit -> {csv_p}")

    summary={"total":len(rows),"verdicts":dict(counts),
        "pct":{k:round(100*v/max(1,len(rows)),1) for k,v in counts.items()}}
    with open(out/"summary.json","w") as f: json.dump(summary,f,indent=2)
    print(json.dumps(summary,indent=2))

    # Bucket analysis
    print("\n=== bucket: cluster_size ===")
    for cs_lo,cs_hi in [(2,2),(3,4),(5,999)]:
        sub=[r for r in rows if cs_lo<=r["cluster_size"]<=cs_hi]
        if not sub: continue
        c=Counter(r["verdict"] for r in sub)
        print(f"  cs={cs_lo}-{cs_hi if cs_hi<999 else '+'}: n={len(sub)} {dict(c)}")

    print("\n=== bucket: frame_density ===")
    for lo,hi in [(0,30),(31,50),(51,80),(81,999)]:
        sub=[r for r in rows if lo<=r["frame_density"]<=hi]
        if not sub: continue
        c=Counter(r["verdict"] for r in sub)
        print(f"  fd={lo}-{hi if hi<999 else '+'}: n={len(sub)} {dict(c)}")

    print("\n=== bucket: row_margin ===")
    for lo,hi in [(0,0.01),(0.011,0.02),(0.021,0.03),(0.031,0.05),(0.051,1.0)]:
        sub=[r for r in rows if lo<=r["row_margin"]<=hi]
        if not sub: continue
        c=Counter(r["verdict"] for r in sub)
        print(f"  rm={lo}-{hi}: n={len(sub)} {dict(c)}")

    print("\n=== bucket: reid_dist ===")
    for lo,hi in [(0.15,0.20),(0.21,0.30),(0.31,0.40),(0.41,1.0)]:
        sub=[r for r in rows if lo<=r["reid_dist"]<=hi]
        if not sub: continue
        c=Counter(r["verdict"] for r in sub)
        print(f"  rd={lo}-{hi}: n={len(sub)} {dict(c)}")

    print("\n=== bucket: v2_gain ===")
    for lo,hi in [(-1.0,-0.05),(-0.05,0.05),(0.05,0.15),(0.15,1.0)]:
        sub=[r for r in rows if lo<=r["v2_gain"]<=hi]
        if not sub: continue
        c=Counter(r["verdict"] for r in sub)
        print(f"  vg={lo}-{hi}: n={len(sub)} {dict(c)}")

    # Beneficial vs harmful feature stats
    print("\n=== feature means by verdict ===")
    for v_name in ["beneficial_trigger","harmful_override","harmful_fn","neutral"]:
        sub=[r for r in rows if r["verdict"]==v_name]
        if not sub: continue
        print(f"\n  {v_name} (n={len(sub)}):")
        for feat in ["cluster_size","frame_density","row_margin","reid_dist","v2_gain","v2_app","v2_motion","v2_dir"]:
            vals=[r[feat] for r in sub if r[feat]>=-0.5]
            if vals:
                print(f"    {feat}: mean={sum(vals)/len(vals):.3f} min={min(vals):.3f} max={max(vals):.3f}")

    # Separability: can we distinguish beneficial from harmful?
    print("\n=== separability analysis ===")
    ben=[r for r in rows if r["verdict"]=="beneficial_trigger"]
    har=[r for r in rows if r["verdict"]=="harmful_override"]
    if ben and har:
        for feat in ["cluster_size","frame_density","row_margin","col_margin","reid_dist","v2_gain","v2_app","v2_motion","v2_dir"]:
            bv=[r[feat] for r in ben if r[feat]>=-0.5]
            hv=[r[feat] for r in har if r[feat]>=-0.5]
            if bv and hv:
                bm_=sum(bv)/len(bv); hm_=sum(hv)/len(hv)
                print(f"  {feat:15s}: ben={bm_:.3f}  har={hm_:.3f}  diff={bm_-hm_:+.3f}")

    # v2_gain threshold sweep
    print("\n=== v2_gain threshold sweep ===")
    print("  (only trigger when v2_gain > threshold)")
    for thr in [-0.1,0.0,0.05,0.10,0.15,0.20]:
        kept=[r for r in rows if r["v2_gain"]>thr]
        ben_k=sum(1 for r in kept if r["verdict"]=="beneficial_trigger")
        har_k=sum(1 for r in kept if r["verdict"]=="harmful_override")
        fn_k=sum(1 for r in kept if r["verdict"]=="harmful_fn")
        neu_k=sum(1 for r in kept if r["verdict"] in ("neutral","neutral_both_wrong"))
        unc_k=sum(1 for r in kept if r["verdict"]=="uncertain")
        print(f"  vg>{thr:.2f}: kept={len(kept):3d} ben={ben_k:3d} har={har_k:3d} fn={fn_k:3d} neu={neu_k:3d} unc={unc_k:3d}")

if __name__=="__main__": main()

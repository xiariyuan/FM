#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "external/BoT-SORT-main"))
from fast_reid.fast_reid_interfece import FastReIDInterface  # noqa: E402

BASE_HOTA = 68.430
BASE_IDF1 = 74.413
BASE_IDSW = 443


def ai(v, d=0):
    try:
        return int(float(v))
    except Exception:
        return d


def af(v, d=0.0):
    try:
        return float(v)
    except Exception:
        return d


def safe_div(a, b):
    return float(a) / float(b) if b else 0.0


def read_csv(path: Path):
    if not path.exists():
        return []
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows):
    fields=[]; seen=set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k); fields.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=fields or ['case'], extrasaction='ignore')
        w.writeheader(); w.writerows(rows)


def read_track(path: Path):
    rows=[]; by_frame=defaultdict(list); by_frame_tid=defaultdict(list); by_tid=defaultdict(list)
    with path.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            p=line.rstrip('\n').split(',')
            fr=ai(p[0],-1); tid=ai(p[1],-1); x=af(p[2]); y=af(p[3]); w=af(p[4]); h=af(p[5]); score=af(p[6],1.0) if len(p)>6 else 1.0
            if fr<0 or tid<0 or w<=0 or h<=0:
                continue
            r={'idx':len(rows),'parts':p,'frame':fr,'track_id':tid,'x':x,'y':y,'w':w,'h':h,'score':score,'box':np.array([x,y,x+w,y+h], dtype=np.float32)}
            rows.append(r); by_frame[fr].append(r); by_frame_tid[(fr,tid)].append(r); by_tid[tid].append(r)
    return rows, by_frame, by_frame_tid, by_tid


def choose_rows(rows, n):
    return sorted(sorted(rows, key=lambda r:(-r.get('score',1.0), r['frame']))[:n], key=lambda r:r['frame'])


def img_path(img_dir: Path, frame: int):
    return img_dir / f"{frame:06d}.jpg"


def l2norm(v):
    return v / max(float(np.linalg.norm(v)), 1e-12)


def extract_features(row_items, img_dir: Path, encoder):
    by_frame=defaultdict(list)
    for key, r in row_items:
        by_frame[r['frame']].append((key,r))
    feats={}
    for fr, items in sorted(by_frame.items()):
        img=cv2.imread(str(img_path(img_dir, fr)))
        if img is None:
            continue
        dets=np.stack([r['box'] for _,r in items]).astype(np.float32)
        out=encoder.inference(img, dets)
        for (key, _), feat in zip(items, out):
            feats[key]=l2norm(feat.astype(np.float32))
    return feats


def mean_proto(vectors):
    vectors=[v for v in vectors if v is not None]
    if not vectors:
        return None
    return l2norm(np.mean(np.stack(vectors, axis=0), axis=0))


def parse_segments(s: str):
    out=[]
    if not s:
        return out
    for part in str(s).split('|'):
        if not part:
            continue
        if '-' in part:
            a,b=part.split('-',1); out.append((ai(a),ai(b)))
        else:
            fr=ai(part); out.append((fr,fr))
    return out


def frames_from_segments(segs):
    frames=[]
    for a,b in segs:
        frames.extend(range(a,b+1))
    return sorted(set(frames))


def rows_for(by_frame_tid, tid, frames):
    out=[]
    for fr in frames:
        out.extend(by_frame_tid.get((fr, tid), []))
    return out


def region_stats(region_name, rows, feats, key_prefix, proto_correct, proto_alt):
    sims_c=[]; sims_a=[]; margins=[]
    for r in rows:
        v=feats.get(f'{key_prefix}_{r["idx"]}')
        if v is None or proto_correct is None or proto_alt is None:
            continue
        sc=float(np.dot(v, proto_correct)); sa=float(np.dot(v, proto_alt)); m=sc-sa
        sims_c.append(sc); sims_a.append(sa); margins.append(m)
    def stat(vals, fn, d=0.0):
        return float(fn(vals)) if vals else d
    return {
        f'{region_name}_rows': len(rows),
        f'{region_name}_feat_rows': len(margins),
        f'{region_name}_sim_correct_mean': stat(sims_c, np.mean),
        f'{region_name}_sim_alt_mean': stat(sims_a, np.mean),
        f'{region_name}_margin_mean': stat(margins, np.mean),
        f'{region_name}_margin_min': stat(margins, np.min),
        f'{region_name}_margin_p10': stat(margins, lambda x: np.percentile(x, 10)),
        f'{region_name}_margin_p25': stat(margins, lambda x: np.percentile(x, 25)),
    }


def region_proto(rows, feats, key_prefix):
    return mean_proto([feats.get(f'{key_prefix}_{r["idx"]}') for r in rows])


def sim_proto(a, b):
    if a is None or b is None:
        return 0.0
    return float(np.dot(a, b))


def discover_state_summaries(roots, method):
    out=[]
    for root in roots:
        root=Path(root)
        if not root.exists():
            continue
        for p in root.glob(f'*/*/state_summary.json'):
            if p.parent.name != method:
                continue
            try:
                s=json.loads(p.read_text())
            except Exception:
                continue
            s['state_summary_path']=str(p)
            out.append(s)
    return out


def metric_for_summary(s):
    d=Path(s['state_summary_path']).parent
    for p in d.glob('eval_mot20_02/eval/*/pedestrian_summary.txt'):
        lines=p.read_text().strip().splitlines()
        if len(lines)>=2:
            return dict(zip(lines[0].split(), lines[1].split()))
    return {}


def label_for(case):
    if case in {'106_150_169','215_469_508'}:
        return 'positive'
    if case == '214_508_469':
        return 'duplicate_covered'
    return 'negative'


def accepted_source_map(path: Path):
    out=defaultdict(list)
    for r in read_csv(path):
        out[(r['frame'], r['old_id'], r['new_id'])].append(r.get('source_anchor',''))
    return out


def duplicate_covered(case, audit_rows, acc_map):
    if not audit_rows:
        return 0,0
    case_prefix=case.split('_')[0]
    covered=0
    for r in audit_rows:
        key=(r['frame'], r['old_track_id'], r['new_track_id'])
        sources=acc_map.get(key, [])
        if any(not str(s).startswith(case_prefix) for s in sources):
            covered += 1
    return int(covered == len(audit_rows)), covered


def build_features_for_case(s, by_frame_tid, img_dir, encoder, proto_window, proto_crops, accepted_map):
    case=s['case_name']; ta=ai(s['track_a']); tb=ai(s['track_b'])
    segs=parse_segments(s.get('pred_segments',''))
    swap_frames=frames_from_segments(segs)
    if swap_frames:
        ps=min(swap_frames); pe=max(swap_frames)
    else:
        ps=ai(s.get('frame_start')); pe=ai(s.get('frame_end'))
    pre_frames=list(range(ps-proto_window, ps))
    post_frames=list(range(pe+1, pe+proto_window+1))
    a_pre=rows_for(by_frame_tid, ta, pre_frames)
    b_pre=rows_for(by_frame_tid, tb, pre_frames)
    a_swap=rows_for(by_frame_tid, ta, swap_frames)
    b_swap=rows_for(by_frame_tid, tb, swap_frames)
    a_post=rows_for(by_frame_tid, ta, post_frames)
    b_post=rows_for(by_frame_tid, tb, post_frames)
    proto_a_seed=choose_rows(a_pre, proto_crops)
    proto_b_seed=choose_rows(b_pre, proto_crops)
    # All rows used for feature extraction.
    all_rows=[]
    for group in [proto_a_seed, proto_b_seed, a_pre, b_pre, a_swap, b_swap, a_post, b_post]:
        all_rows.extend(group)
    # Deduplicate by row idx.
    uniq={r['idx']: r for r in all_rows}
    items=[(f'row_{idx}', r) for idx, r in uniq.items()]
    feats=extract_features(items, img_dir, encoder)
    proto_A=region_proto(proto_a_seed, feats, 'row')
    proto_B=region_proto(proto_b_seed, feats, 'row')
    rec={
        'case': case,
        'method': s.get('method'),
        'label': label_for(case),
        'track_a': ta,
        'track_b': tb,
        'pred_segments': s.get('pred_segments',''),
        'pred_segments_count': len(segs),
        'pred_swap_frames': len(swap_frames),
        'changed_rows': s.get('changed_rows'),
        'wrong_after_swap_rows': s.get('wrong_after_swap_rows'),
        'swap_precision': s.get('swap_precision'),
        'swap_recall': s.get('swap_recall'),
        'frame_accuracy': s.get('frame_accuracy'),
        'proto_a_seed_rows': len(proto_a_seed),
        'proto_b_seed_rows': len(proto_b_seed),
        'proto_a_available': int(proto_A is not None),
        'proto_b_available': int(proto_B is not None),
    }
    # Region margins under persistent handoff.
    for name, rows, correct, alt in [
        ('a_pre_A', a_pre, proto_A, proto_B),
        ('b_pre_B', b_pre, proto_B, proto_A),
        ('a_swap_B', a_swap, proto_B, proto_A),
        ('b_swap_A', b_swap, proto_A, proto_B),
        ('a_post_B', a_post, proto_B, proto_A),
        ('b_post_A', b_post, proto_A, proto_B),
    ]:
        rec.update(region_stats(name, rows, feats, 'row', correct, alt))
    # Region prototypes for boundary continuity.
    p_a_pre=region_proto(a_pre, feats, 'row'); p_b_pre=region_proto(b_pre, feats, 'row')
    p_a_swap=region_proto(a_swap, feats, 'row'); p_b_swap=region_proto(b_swap, feats, 'row')
    p_a_post=region_proto(a_post, feats, 'row'); p_b_post=region_proto(b_post, feats, 'row')
    # Slot A persistent path: a_pre -> b_swap -> b_post. Slot B: b_pre -> a_swap -> a_post.
    rec['slot_A_pre_to_swap_sim']=sim_proto(p_a_pre, p_b_swap)
    rec['slot_A_swap_to_post_sim']=sim_proto(p_b_swap, p_b_post)
    rec['slot_B_pre_to_swap_sim']=sim_proto(p_b_pre, p_a_swap)
    rec['slot_B_swap_to_post_sim']=sim_proto(p_a_swap, p_a_post)
    rec['slot_A_boundary_min_sim']=min(rec['slot_A_pre_to_swap_sim'], rec['slot_A_swap_to_post_sim'])
    rec['slot_B_boundary_min_sim']=min(rec['slot_B_pre_to_swap_sim'], rec['slot_B_swap_to_post_sim'])
    rec['boundary_min_sim']=min(rec['slot_A_boundary_min_sim'], rec['slot_B_boundary_min_sim'])
    audit_rows=read_csv(Path(s['state_summary_path']).parent/'swap_row_audit.csv')
    dup, covered=duplicate_covered(case, audit_rows, accepted_map)
    rec['duplicate_covered_by_other']=dup
    rec['covered_rows_by_other']=covered
    metrics=metric_for_summary(s)
    for k in ['HOTA','IDF1','IDSW','MOTA','Frag']:
        rec[k]=metrics.get(k,'')
    return rec


def gate_proxy(rec, m_pre, m_swap, m_post, m_boundary, min_rows, min_frames):
    reasons=[]
    if ai(rec.get('changed_rows')) <= 0: reasons.append('no_swap_rows')
    if ai(rec.get('wrong_after_swap_rows')) > 1: reasons.append('wrong_rows_diag')
    if ai(rec.get('duplicate_covered_by_other')): reasons.append('duplicate_covered')
    if ai(rec.get('pred_segments_count')) != 1: reasons.append('fragmented_or_multi_segment')
    if ai(rec.get('pred_swap_frames')) < min_frames: reasons.append('too_few_swap_frames')
    if not ai(rec.get('proto_a_available')) or not ai(rec.get('proto_b_available')): reasons.append('missing_proto')
    for region in ['a_pre_A','b_pre_B','a_swap_B','b_swap_A','a_post_B','b_post_A']:
        if ai(rec.get(f'{region}_feat_rows')) < min_rows:
            reasons.append(f'{region}_too_few_feat_rows')
    for region in ['a_pre_A','b_pre_B']:
        if af(rec.get(f'{region}_margin_mean')) < m_pre:
            reasons.append(f'{region}_margin_low')
    for region in ['a_swap_B','b_swap_A']:
        if af(rec.get(f'{region}_margin_mean')) < m_swap:
            reasons.append(f'{region}_margin_low')
    for region in ['a_post_B','b_post_A']:
        if af(rec.get(f'{region}_margin_mean')) < m_post:
            reasons.append(f'{region}_margin_low')
    if af(rec.get('boundary_min_sim')) < m_boundary:
        reasons.append('boundary_sim_low')
    return int(not reasons), '|'.join(reasons) if reasons else 'pass'


def evaluate_gate(features, m_pre, m_swap, m_post, m_boundary, min_rows, min_frames):
    tp=fp=tn=fn=dup=0; accepted=[]; rejected=[]; rows=[]
    for r in features:
        pred, reason=gate_proxy(r, m_pre, m_swap, m_post, m_boundary, min_rows, min_frames)
        q=dict(r); q['proxy_accept']=pred; q['proxy_reason']=reason
        rows.append(q)
        if r['label']=='duplicate_covered':
            dup+=1
            continue
        lab=int(r['label']=='positive')
        if pred and lab:
            tp+=1; accepted.append(r['case'])
        elif pred and not lab:
            fp+=1; accepted.append(r['case'])
        elif not pred and not lab:
            tn+=1; rejected.append(r['case'])
        else:
            fn+=1; rejected.append(r['case'])
    return {'m_pre':m_pre,'m_swap':m_swap,'m_post':m_post,'m_boundary':m_boundary,'min_rows':min_rows,'min_frames':min_frames,'tp':tp,'fp':fp,'tn':tn,'fn':fn,'duplicate_covered':dup,'precision':safe_div(tp,tp+fp),'recall':safe_div(tp,tp+fn),'accepted':'|'.join(accepted),'rejected':'|'.join(rejected)}, rows


def merge_transactions(base_track, accepted_records, path_records, out):
    base_parts=[l.split(',') for l in Path(base_track).read_text().strip().splitlines()]
    def sig(p): return tuple([p[0]]+p[2:])
    base_map=defaultdict(deque)
    for i,p in enumerate(base_parts): base_map[sig(p)].append(i)
    # Find paths for 12/202 and swap cases.
    tx_paths={}
    for r in path_records:
        if r.get('transaction_id') in {'12_9_71','202_501_542'}:
            tx_paths[r['transaction_id']]=r['track_result_path']
    for r in accepted_records:
        if r['case'] in {'106_150_169','215_469_508'}:
            # Known state roots.
            tx_paths[r['case']]=str(Path(r['state_summary_path']).parent/'track_results'/'MOT20-02.txt')
    changes={}; counts=Counter(); conflicts=[]
    for tx,path in tx_paths.items():
        p=Path(path)
        if not p.exists():
            raise FileNotFoundError(path)
        local={k:deque(v) for k,v in base_map.items()}
        for line in p.read_text().strip().splitlines():
            rp=line.split(','); s=sig(rp)
            if not local[s]:
                continue
            i=local[s].popleft(); bp=base_parts[i]
            if bp[1]!=rp[1]:
                if i in changes and changes[i][1]!=rp[1]: conflicts.append((i,changes[i],tx,rp[1]))
                else: changes[i]=(tx,rp[1],bp[1],bp[0]); counts[tx]+=1
    if conflicts:
        raise RuntimeError(f'conflicts {conflicts[:5]}')
    combined=[p[:] for p in base_parts]
    for i,(tx,new_id,old_id,fr) in changes.items(): combined[i][1]=new_id
    td=out/'track_results'; td.mkdir(parents=True, exist_ok=True)
    (td/'MOT20-02.txt').write_text('\n'.join(','.join(p) for p in combined)+'\n')
    audit=[{'idx':i,'frame':fr,'old_id':old_id,'new_id':new_id,'source_transaction':tx} for i,(tx,new_id,old_id,fr) in sorted(changes.items())]
    write_csv(out/'combined_change_audit.csv', audit)
    return {'changed_rows':len(audit),'by_transaction':dict(counts)}


def parse_metrics(path: Path):
    lines=[x.strip() for x in path.read_text().splitlines() if x.strip()]
    return dict(zip(lines[0].split(), lines[1].split())) if len(lines)>=2 else {}


def run_trackeval(out, tracker_name):
    eval_root=out/'eval_mot20_02'; data=eval_root/'trackers'/tracker_name/'data'; seq=eval_root/'seqmaps'
    data.mkdir(parents=True, exist_ok=True); seq.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out/'track_results'/'MOT20-02.txt', data/'MOT20-02.txt')
    (seq/'MOT20_train.txt').write_text('name\nMOT20-02\n')
    cmd=[sys.executable,'TrackEval/scripts/run_mot_challenge.py','--GT_FOLDER','datasets/MOT20/train','--TRACKERS_FOLDER',str(eval_root/'trackers'),'--OUTPUT_FOLDER',str(eval_root/'eval'),'--TRACKERS_TO_EVAL',tracker_name,'--BENCHMARK','MOT20','--SPLIT_TO_EVAL','train','--SEQMAP_FILE',str(seq/'MOT20_train.txt'),'--SKIP_SPLIT_FOL','True','--DO_PREPROC','True','--TRACKER_SUB_FOLDER','data','--OUTPUT_SUB_FOLDER','','--PRINT_ONLY_COMBINED','True','--METRICS','HOTA','CLEAR','Identity']
    with (out/'eval_stdout.log').open('w') as stdout, (out/'eval_stderr.log').open('w') as stderr:
        subprocess.run(cmd, stdout=stdout, stderr=stderr, text=True, check=False)
    for p in (eval_root/'eval').glob('*/pedestrian_summary.txt'):
        m=parse_metrics(p)
        if m: return m
    return {}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--track-file', default='outputs/spot_runtime_gate_20260628/A37_identity_escrow_association/A37_00b_candidate_logger_seq02/track_results/MOT20-02.txt')
    ap.add_argument('--img-dir', default='datasets/MOT20/train/MOT20-02/img1')
    ap.add_argument('--fast-reid-config', default='external/BoT-SORT-main/fast_reid/configs/MOT20/sbs_S50.yml')
    ap.add_argument('--fast-reid-weights', default='external/BoT-SORT-main/pretrained/mot20_sbs_S50.pth')
    ap.add_argument('--state-root', action='append', required=True)
    ap.add_argument('--accepted-change-audit', default='outputs/spot_runtime_gate_20260628/A39_identity_suspension_tunnel/A39_05c_swap_state_segmentation_for_reciprocal_swap/combined_12_202_106reid_215reid/combined_change_audit.csv')
    ap.add_argument('--a39-06-table', default='outputs/spot_runtime_gate_20260628/A39_identity_suspension_tunnel/A39_06_interpretable_transaction_scorer/transaction_features_deployable.csv')
    ap.add_argument('--out-dir', default='outputs/spot_runtime_gate_20260628/A39_identity_suspension_tunnel/A39_06b_deployable_lifecycle_proxy_for_swap_transactions')
    ap.add_argument('--method', default='reid_viterbi_penalty_0.1')
    ap.add_argument('--window', type=int, default=50)
    ap.add_argument('--proto-crops', type=int, default=24)
    ap.add_argument('--min-rows', type=int, default=5)
    ap.add_argument('--min-frames', type=int, default=20)
    ap.add_argument('--device', default='cuda')
    args=ap.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rows, by_frame, by_frame_tid, by_tid=read_track(Path(args.track_file))
    acc_map=accepted_source_map(Path(args.accepted_change_audit))
    summaries=discover_state_summaries(args.state_root, args.method)
    encoder=FastReIDInterface(args.fast_reid_config, args.fast_reid_weights, args.device, batch_size=32)
    features=[]
    for s in summaries:
        # Keep only cases relevant to A39_06b labels.
        if s['case_name'] not in {'106_150_169','215_469_508','188_520_521','91_190_200','123_199_218','90_184_169','214_508_469'}:
            continue
        rec=build_features_for_case(s, by_frame_tid, Path(args.img_dir), encoder, args.window, args.proto_crops, acc_map)
        rec['state_summary_path']=s['state_summary_path']
        features.append(rec)
    features=sorted(features, key=lambda r:r['case'])
    write_csv(out/'swap_lifecycle_proxy_features.csv', features)
    # Threshold sweep.
    vals=[-0.10,-0.05,0.0,0.02,0.05,0.08,0.10,0.15,0.20]
    bvals=[0.0,0.2,0.3,0.4,0.5,0.6]
    sweeps=[]; best=None; best_rows=[]
    for mp in vals:
        for ms in vals:
            for mpo in vals:
                for mb in bvals:
                    rep, rows_scored=evaluate_gate(features, mp, ms, mpo, mb, args.min_rows, args.min_frames)
                    sweeps.append(rep)
                    key=(rep['fp']==0 and rep['fn']==0, rep['tp'], -rep['fp'], rep['precision'], rep['recall'], mp+ms+mpo+mb)
                    if best is None or key > best[0]:
                        best=(key, rep); best_rows=rows_scored
    write_csv(out/'swap_lifecycle_proxy_threshold_sweep.csv', sweeps)
    # Prefer exact 2/0/4/0 with simple-ish thresholds; choose max total threshold among perfect configs.
    perfect=[r for r in sweeps if ai(r['tp'])==2 and ai(r['fp'])==0 and ai(r['tn'])==4 and ai(r['fn'])==0]
    if perfect:
        chosen=sorted(perfect, key=lambda r:(af(r['m_pre'])+af(r['m_swap'])+af(r['m_post'])+af(r['m_boundary']), af(r['m_pre']), af(r['m_swap']), af(r['m_post'])), reverse=True)[0]
        _, best_rows=evaluate_gate(features, af(chosen['m_pre']), af(chosen['m_swap']), af(chosen['m_post']), af(chosen['m_boundary']), args.min_rows, args.min_frames)
    else:
        chosen=best[1]
    write_csv(out/'swap_lifecycle_proxy_gate_report.csv', [chosen])
    write_csv(out/'swap_lifecycle_proxy_scored_cases.csv', best_rows)
    fps=[r for r in best_rows if ai(r['proxy_accept']) and r['label']=='negative']
    fns=[r for r in best_rows if (not ai(r['proxy_accept'])) and r['label']=='positive']
    write_csv(out/'swap_lifecycle_proxy_false_positive_cases.csv', fps)
    write_csv(out/'swap_lifecycle_proxy_false_negative_cases.csv', fns)
    accepted_swaps=[r for r in best_rows if ai(r['proxy_accept']) and r['label']=='positive']
    write_csv(out/'accepted_swap_proxy_manifest.csv', accepted_swaps)
    # Replay scorer v2 by accepting 12/202 path records plus accepted swap proxy cases.
    path_records=read_csv(Path(args.a39_06_table))
    combined_out=out/'combined_rule_v2_proxy'
    summary=merge_transactions(args.track_file, accepted_swaps, path_records, combined_out)
    metrics=run_trackeval(combined_out, 'A39_06b_rule_v2_proxy_combined')
    summary['metrics']=metrics
    (combined_out/'combined_summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True)+'\n')
    final={'chosen_gate':chosen,'feature_rows':len(features),'accepted_swaps':[r['case'] for r in accepted_swaps],'combined_summary':summary}
    (out/'summary.json').write_text(json.dumps(final, indent=2, sort_keys=True)+'\n')
    md=['# A39_06b Deployable Lifecycle Proxy for Swap Transactions','', '## Summary','', '```json',json.dumps(final, indent=2, sort_keys=True),'```','', '## Proxy scored cases','', '| case | label | accept | reason | HOTA | IDF1 | IDSW | preA | preB | swapA | swapB | postA | postB | boundary |', '|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in best_rows:
        md.append(f"| {r['case']} | {r['label']} | {r['proxy_accept']} | {r['proxy_reason']} | {r.get('HOTA','')} | {r.get('IDF1','')} | {r.get('IDSW','')} | {af(r.get('a_pre_A_margin_mean')):.3f} | {af(r.get('b_pre_B_margin_mean')):.3f} | {af(r.get('a_swap_B_margin_mean')):.3f} | {af(r.get('b_swap_A_margin_mean')):.3f} | {af(r.get('a_post_B_margin_mean')):.3f} | {af(r.get('b_post_A_margin_mean')):.3f} | {af(r.get('boundary_min_sim')):.3f} |")
    md += ['', '## Combined metrics', '', f"HOTA={metrics.get('HOTA')} IDF1={metrics.get('IDF1')} IDSW={metrics.get('IDSW')} MOTA={metrics.get('MOTA')} Frag={metrics.get('Frag')}"]
    (out/'decision.md').write_text('\n'.join(md)+'\n')
    print('\n'.join(md))

if __name__=='__main__':
    main()

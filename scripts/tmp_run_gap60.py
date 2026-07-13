#!/usr/bin/env python3
import subprocess, sys
cmd=[sys.executable,'scripts/postprocess/oracle_tracklet_link_bound.py','--input-dir','outputs/alink_train_inputs/parambest_track_results','--gt-root','/gemini/code/datasets/MOT20/train','--output-dir','outputs/oracle_link_bound_train/gap60/linked_results','--seqs','MOT20-01','MOT20-02','MOT20-03','MOT20-05','--iou-thr','0.5','--max-gap','60','--min-purity','0.60','--min-match-frac','0.20','--min-majority-count','2','--summary-json','outputs/oracle_link_bound_train/gap60/oracle_summary.json','--summary-csv','outputs/oracle_link_bound_train/gap60/oracle_summary.csv']
subprocess.run(cmd,check=True)

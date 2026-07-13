#!/usr/bin/env python3
import subprocess, sys
base=['scripts/postprocess/oracle_tracklet_link_bound.py','--input-dir','outputs/alink_train_inputs/parambest_track_results','--gt-root','/gemini/code/datasets/MOT20/train','--seqs','MOT20-01','MOT20-02','MOT20-03','MOT20-05','--iou-thr','0.5','--min-purity','0.60','--min-match-frac','0.20','--min-majority-count','2']
cmd=[sys.executable]+base+['--max-gap','999999','--output-dir','outputs/oracle_link_bound_train/global/linked_results','--summary-json','outputs/oracle_link_bound_train/global/oracle_summary.json','--summary-csv','outputs/oracle_link_bound_train/global/oracle_summary.csv']
subprocess.run(cmd,check=True)

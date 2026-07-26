from __future__ import annotations
import argparse,importlib.util,sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[2]
SEQS=('MOT20-02','MOT20-03','MOT20-05')
def load_impl(seq):
 p=REPO/'scripts/m28_research/m28_a3_strong_host_deferred_identity_capacity.py';spec=importlib.util.spec_from_file_location(f'm28_a4_impl_{seq[-2:]}',p);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
 m.SEQ=seq
 m.BASELINE=REPO/f'outputs/mot20_m23_20260718/m23_46_pure_hgb_topk_combined_oof_v1/track_results/{seq}.txt'
 m.DUMP=REPO/f'outputs/alink_train_inputs/phase0_root/{seq}/dump_yolox_reid.npz'
 m.ROOT=REPO/f'outputs/mot20_m28_20260726/m28_a4_strong_host_multisequence/{seq}'
 return m
def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=('freeze','teacher'));p.add_argument('--seq',required=True,choices=SEQS);a=p.parse_args();m=load_impl(a.seq);m.ROOT.mkdir(parents=True,exist_ok=True);m.freeze() if a.stage=='freeze' else m.teacher()
if __name__=='__main__':main()

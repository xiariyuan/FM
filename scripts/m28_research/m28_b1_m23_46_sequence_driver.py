from __future__ import annotations
import argparse,importlib.util,sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[2]
ROOT=REPO/'outputs/mot20_m28_20260726/m28_b1_m23_46_multisequence'
BASE=REPO/'outputs/mot20_m23_20260718/m23_46_pure_hgb_topk_combined_oof_v1/track_results'
DUMPS=REPO/'outputs/alink_train_inputs/phase0_root'
SEQS=('MOT20-01','MOT20-02','MOT20-03','MOT20-05')
def load(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def run(seq,stage):
 m=load(f'm28b1_{stage}_{seq[-2:]}',REPO/'scripts/m28_research/m28_b0_m23_46_deferred_identity.py');m.SEQ=seq;m.ROOT=ROOT/seq;m.BASELINE=BASE/f'{seq}.txt';m.DUMP=DUMPS/seq/'dump_yolox_reid.npz';m.ROOT.mkdir(parents=True,exist_ok=True)
 if stage=='freeze-candidates':m.freeze_candidates()
 else:m.teacher()
def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze-candidates','teacher']);p.add_argument('--seq',required=True,choices=SEQS);a=p.parse_args();run(a.seq,a.stage)
if __name__=='__main__':main()

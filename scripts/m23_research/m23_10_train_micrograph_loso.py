from __future__ import annotations

# Research artifact for the MOT20 M23 sequence-LOSO audit.
import json,math
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score,roc_auc_score

SEQS=['MOT20-01','MOT20-02','MOT20-03','MOT20-05']
ROOT=Path('outputs/mot20_m23_20260718/micrograph_chunk30_v1')
OUT=Path('outputs/mot20_m23_20260718/micrograph_chunk30_loso_v1')
FEATURES=['gap','log_gap','appearance_cos','same_source','source_adjacent','forward_motion_error','backward_motion_error','motion_error_min','motion_error_mean','endpoint_displacement','velocity_cos','log_height_ratio','src_rows','dst_rows','src_mapping_rate','dst_mapping_rate','mapping_rate_min','src_consistency','dst_consistency','consistency_min','src_match_iou','dst_match_iou','out_rank','in_rank','max_rank','out_margin','in_margin','max_margin']

def seq_weights(frame):
 c=frame.seq.value_counts();return frame.seq.map({s:len(frame)/(len(c)*n) for s,n in c.items()}).to_numpy(float)

def safe_metrics(y,p):
 if len(np.unique(y))<2:return {'auc':None,'ap':None}
 return {'auc':float(roc_auc_score(y,p)),'ap':float(average_precision_score(y,p))}

def main():
 OUT.mkdir(parents=True,exist_ok=True);frames=[]
 for seq in SEQS:
  d=pd.read_parquet(ROOT/seq/'candidate_edges.parquet');d.insert(0,'seq',seq);frames.append(d)
 all_data=pd.concat(frames,ignore_index=True,sort=False);folds=[]
 for held in SEQS:
  train=all_data[(all_data.seq!=held)&(all_data.src_modal_gt>0)&(all_data.dst_modal_gt>0)&(all_data.src_purity>=.7)&(all_data.dst_purity>=.7)].copy()
  test=all_data[all_data.seq==held].copy();clean=test[(test.src_modal_gt>0)&(test.dst_modal_gt>0)&(test.src_purity>=.7)&(test.dst_purity>=.7)].copy()
  model=HistGradientBoostingClassifier(max_iter=240,learning_rate=.05,max_leaf_nodes=31,min_samples_leaf=40,l2_regularization=4.0,max_bins=255,random_state=10100+SEQS.index(held),early_stopping=False)
  model.fit(train[FEATURES],train.same_gt.astype(int),sample_weight=seq_weights(train)*train.label_confidence.clip(.2,1.0).to_numpy(float))
  test['pred_same_prob']=model.predict_proba(test[FEATURES])[:,1];test['held_out_seq']=held;test.to_parquet(OUT/f'{held}_edge_predictions.parquet',index=False)
  clean_p=test.loc[clean.index,'pred_same_prob'].to_numpy();y=clean.same_gt.to_numpy(int);rec={'held_out_seq':held,'train_rows':len(train),'train_positive':int(train.same_gt.sum()),'test_edges':len(test),'clean_test_rows':len(clean),'clean_test_positive':int(y.sum()),**safe_metrics(y,clean_p)}
  for subset_name,mask in [('source_adjacent',clean.source_adjacent==1),('cross',clean.same_source==0)]:
   yy=clean.loc[mask,'same_gt'].to_numpy(int);pp=test.loc[clean.loc[mask].index,'pred_same_prob'].to_numpy();met=safe_metrics(yy,pp);rec[f'{subset_name}_rows']=len(yy);rec[f'{subset_name}_positive']=int(yy.sum());rec[f'{subset_name}_auc']=met['auc'];rec[f'{subset_name}_ap']=met['ap']
   for t in [.1,.3,.5,.7,.9]:
    sel=pp>=t;rec[f'{subset_name}_t{int(t*10):02d}_selected']=int(sel.sum());rec[f'{subset_name}_t{int(t*10):02d}_precision']=float(yy[sel].mean()) if sel.any() else None;rec[f'{subset_name}_t{int(t*10):02d}_recall']=float(yy[sel].sum()/max(yy.sum(),1))
  folds.append(rec);print(json.dumps(rec),flush=True)
 report={'protocol':{'validation':'strict leave-one-sequence-out; held sequence excluded from fitting and preprocessing','model':'HistGradientBoostingClassifier fixed hyperparameters','features':FEATURES,'train_labels':'modal-GT equality on clean chunks; GT columns excluded from features','inference':'GT-free'},'folds':folds,'dataset':{'edges':len(all_data),'positive':int(all_data.same_gt.sum())}};(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()

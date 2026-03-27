cd /gemini/code/FMtrack-main/FM-Track
DATA=/gemini/code/datasets
OUT=./outputs/fa_mot_v3_smoke
for e in 3 4 5 6 7; do
  CKPT=$OUT/checkpoint_${e}.pth
  RUN=$OUT/quick_val_epoch${e}
  mkdir -p $RUN
  python submit_public.py \
    --config-path configs/r50_dino_fa_mot_v3_robust.yaml \
    --inference-model $CKPT \
    --inference-mode evaluate \
    --inference-dataset MOT17 \
    --inference-split train \
    --det-thresh 0.2 \
    --newborn-thresh -1 \
    --id-thresh 0.1 \
    --iou-thresh 0.2 \
    --area-thresh 0 \
    --min-track-len 0 \
    --outputs-dir $RUN \
    --data-root $DATA | tee $RUN/metrics.txt
done
grep -R "HOTA = " -n $OUT/quick_val_epoch*/metrics.txt
grep -R "IDF1 = " -n $OUT/quick_val_epoch*/metrics.txt

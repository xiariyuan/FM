#!/bin/bash
cd /gemini/code/FMtrack-main/FM-Track
CONFIG=configs/bytetrack_fa_mot_mot17_v9_fromscratch_unfreezeproj_longcurr_p0fix.yaml
OUT_BASE=outputs/bytetrack_fa_mot_mot17_v9_fromscratch_unfreezeproj_longcurr_p0fix/val_mot20

for EPOCH in 6 7; do
    CKPT=outputs/bytetrack_fa_mot_mot17_v9_fromscratch_unfreezeproj_longcurr_p0fix/checkpoint_epoch_${EPOCH}.pth
    OUT_DIR=${OUT_BASE}/epoch_${EPOCH}
    
    if [ -f "${OUT_DIR}/tracker/MOT20-train/pedestrian_summary.txt" ]; then
        echo "Epoch ${EPOCH} already done, skipping"
        continue
    fi
    
    mkdir -p ${OUT_DIR}
    echo "=== MOT20 val epoch ${EPOCH} ==="
    /root/miniconda3/bin/python3.11 -u submit_bytetrack.py         --config-path ${CONFIG}         --inference-model ${CKPT}         --inference-dataset MOT20         --inference-split train         --output-dir ${OUT_DIR}         --detector-profile mot20         --val-sequences MOT20-01,MOT20-02 2>&1 | tee ${OUT_DIR}/eval.log
    
    echo "=== Epoch ${EPOCH} done ==="
done
echo "All MOT20 validation done"

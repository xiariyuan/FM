#\!/bin/bash
cd /gemini/code/FMtrack-main/FM-Track
CONFIG=configs/bytetrack_fa_mot_mot17_v9_fromscratch_unfreezeproj_longcurr_p0fix.yaml
CKPT=outputs/bytetrack_fa_mot_mot17_v9_fromscratch_unfreezeproj_longcurr_p0fix/checkpoint_epoch_3.pth
BASE=outputs/bytetrack_fa_mot_mot17_v9_fromscratch_unfreezeproj_longcurr_p0fix/val_mot20_sweep

for ID_THRESH in 0.003 0.01 0.05; do
for IOU_GATE in 0.2 0.3; do
for DET_MAX in 50 80; do
    TAG="id${ID_THRESH}_iou${IOU_GATE}_det${DET_MAX}"
    OUT_DIR="${BASE}/${TAG}"
    if [ -f "${OUT_DIR}/tracker/MOT20-train/pedestrian_summary.txt" ]; then
        echo "${TAG} done, skipping"
        continue
    fi
    mkdir -p "${OUT_DIR}"
    echo "=== ${TAG} ==="
    /root/miniconda3/bin/python3.11 -u submit_bytetrack.py         --config-path ${CONFIG}         --inference-model ${CKPT}         --inference-dataset MOT20         --inference-split train         --output-dir "${OUT_DIR}"         --detector-profile sw_yolox_mot20         --val-sequences MOT20-01,MOT20-02         --id-thresh ${ID_THRESH}         --assoc-iou-gate ${IOU_GATE}         --det-max-per-frame ${DET_MAX} 2>&1 | tail -5
    echo "=== ${TAG} done ==="
done
done
done
echo "All sweep done"

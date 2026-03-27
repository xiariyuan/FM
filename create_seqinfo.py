import os

mot17_train_info = {
    "MOT17-02": {"seqLength": 600, "imWidth": 1920, "imHeight": 1080, "frameRate": 30},
    "MOT17-04": {"seqLength": 1050, "imWidth": 1920, "imHeight": 1080, "frameRate": 30},
    "MOT17-05": {"seqLength": 837, "imWidth": 640, "imHeight": 480, "frameRate": 14},
    "MOT17-09": {"seqLength": 525, "imWidth": 1920, "imHeight": 1080, "frameRate": 30},
    "MOT17-10": {"seqLength": 654, "imWidth": 1920, "imHeight": 1080, "frameRate": 30},
    "MOT17-11": {"seqLength": 900, "imWidth": 1920, "imHeight": 1080, "frameRate": 30},
    "MOT17-13": {"seqLength": 750, "imWidth": 1920, "imHeight": 1080, "frameRate": 25},
}

detectors = ["DPM", "FRCNN", "SDP"]
base_path = "/gemini/code/datasets/MOT17/train"

for seq_base, info in mot17_train_info.items():
    for det in detectors:
        seq_name = f"{seq_base}-{det}"
        seq_path = os.path.join(base_path, seq_name)
        ini_path = os.path.join(seq_path, "seqinfo.ini")
        
        if os.path.exists(seq_path) and not os.path.exists(ini_path):
            content = f"""[Sequence]
name={seq_name}
imDir=img1
frameRate={info['frameRate']}
seqLength={info['seqLength']}
imWidth={info['imWidth']}
imHeight={info['imHeight']}
imExt=.jpg
"""
            with open(ini_path, 'w') as f:
                f.write(content)
            print(f"Created: {ini_path}")

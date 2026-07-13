#!/usr/bin/env python3
from __future__ import annotations
import configparser, json
from pathlib import Path

DATA = Path('/gemini/code/datasets/MOT20/train')
OUT = Path('/gemini/code/FMtrack-main/FM-Track/outputs/detector_plugins/dfine_mot20_finetune_data')
IMG_ROOT = DATA
ANN_DIR = OUT / 'annotations'
ANN_DIR.mkdir(parents=True, exist_ok=True)

def seq_meta(seq_dir: Path):
    cp = configparser.ConfigParser()
    cp.read(seq_dir / 'seqinfo.ini')
    s=cp['Sequence']
    return int(s['seqLength']), int(s['imWidth']), int(s['imHeight'])

def start_frame_for_half_val(n: int) -> int:
    return n // 2 + 2

def build(split: str):
    images=[]; anns=[]; img_id=1; ann_id=1; per_seq={}
    for seq_dir in sorted(DATA.glob('MOT20-*')):
        n,W,H=seq_meta(seq_dir)
        val_start=start_frame_for_half_val(n)
        if split=='train_firsthalf':
            use=lambda f: f < val_start
        elif split=='val_secondhalf':
            use=lambda f: f >= val_start
        else:
            raise ValueError(split)
        gt=seq_dir/'gt'/'gt.txt'; img_dir=seq_dir/'img1'
        rows=[]; frames=set()
        for line in gt.read_text().splitlines():
            if not line.strip(): continue
            vals=line.split(',')
            frame=int(float(vals[0])); x,y,w,h=[float(v) for v in vals[2:6]]
            mark=int(float(vals[6])) if len(vals)>6 else 1
            cls=int(float(vals[7])) if len(vals)>7 else 1
            vis=float(vals[8]) if len(vals)>8 else 1.0
            if mark!=1 or cls!=1 or not use(frame) or w<=1 or h<=1:
                continue
            rows.append((frame,x,y,w,h,vis)); frames.add(frame)
        frame_to_imgid={}; seq_img=seq_ann=0
        for frame in sorted(frames):
            fn=f'{frame:06d}.jpg'
            if not (img_dir/fn).exists():
                continue
            frame_to_imgid[frame]=img_id
            images.append({'id':img_id,'file_name':f'{seq_dir.name}/img1/{fn}','width':W,'height':H,'seq':seq_dir.name,'frame_id':frame})
            img_id+=1; seq_img+=1
        for frame,x,y,w,h,vis in rows:
            iid=frame_to_imgid.get(frame)
            if iid is None: continue
            x=max(0.0,min(x,W-1.0)); y=max(0.0,min(y,H-1.0)); w=max(1.0,min(w,W-x)); h=max(1.0,min(h,H-y))
            anns.append({'id':ann_id,'image_id':iid,'category_id':0,'bbox':[round(x,2),round(y,2),round(w,2),round(h,2)],'area':round(w*h,2),'iscrowd':0,'visibility':vis})
            ann_id+=1; seq_ann+=1
        per_seq[seq_dir.name]={'images':seq_img,'annotations':seq_ann,'seqLength':n,'val_start':val_start}
    coco={'info':{'description':f'MOT20 {split} for D-FINE person fine-tune'},'licenses':[],'images':images,'annotations':anns,'categories':[{'id':0,'name':'person','supercategory':'person'}]}
    out=ANN_DIR/f'{split}.json'; out.write_text(json.dumps(coco,separators=(',',':')))
    return out,per_seq,len(images),len(anns)
manifest={}
for split in ['train_firsthalf','val_secondhalf']:
    out,per_seq,ni,na=build(split)
    manifest[split]={'ann_file':str(out),'img_folder':str(IMG_ROOT),'images':ni,'annotations':na,'per_seq':per_seq}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
print(json.dumps(manifest,indent=2,ensure_ascii=False))

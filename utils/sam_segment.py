import torch

from ultralytics import FastSAM, SAM
from ultralytics.models.fastsam import FastSAMPrompt

def sam_segment_object(image_path, output_dir, sam_ckpt, point_coord_sets, point_label_sets ):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # 选择MODEL segment anything
    if sam_ckpt != "./weights/mobile_sam.pt":
        model = FastSAM(sam_ckpt)
        # retina_masks 要求masks是高分辨率的，为了准确性
        results = model(image_path,device=DEVICE,retina_masks=True,imgsz=1024,conf=0.4,iou=0.9)
        prompt_process = FastSAMPrompt(image_path, results, device=DEVICE)
        # Point prompt
        # points default [[0,0]] [[x1,y1],[x2,y2]]
        # point_label default [0] [1,0] 0:background, 1:foreground
        ann = prompt_process.point_prompt(points=point_coord_sets, pointlabel=point_label_sets)
        prompt_process.plot(annotations=ann, output=output_dir)
        
    else:
        model = SAM(sam_ckpt)
import torch
import sys
import argparse
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

from ultralytics import FastSAM, SAM
from ultralytics.models.fastsam import FastSAMPrompt
# from lama_inpaint import inpaint_img_with_lama
from utils import get_clicked_point, get_box_point, get_brush_point


def setup_args(parser):
    parser.add_argument(
        "--input_img", type=str, required=True,
        help="Path to a single input img",
    )
    parser.add_argument(
        "--coords_type", type=str, required=True,
        default="key_in", choices=["click", "box", "brush","key_in"], 
        help="The way to select coords",
    )
    parser.add_argument(
        "--point_coords", type=int, nargs='+', required=True,
        help="The coordinate of the point prompt, [coord_W coord_H].",
    )
    parser.add_argument(
        "--point_labels", type=int, nargs='+', required=True,
        help="The labels of the point prompt, 1 or 0.",
    )
    parser.add_argument(
        "--dilate_kernel_size", type=int, default=None,
        help="Dilate kernel size. Default: None",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output path to the directory with results.",
    )
    parser.add_argument(
        "--sam_model_type", type=str,
        default="FastSAM-s", choices=['FastSAM-s', 'FastSAM-x', 'mobile_sam'],
        help="The type of sam model to load. Default: 'FastSAM-s"
    )
    parser.add_argument(
        "--sam_ckpt", type=str, required=True,
        help="The path to the SAM checkpoint to use for mask generation.",
    )
    # parser.add_argument(
    #     "--lama_config", type=str,
    #     default="./lama/configs/prediction/default.yaml",
    #     help="The path to the config file of lama model. "
    #          "Default: the config of big-lama",
    # )
    # parser.add_argument(
    #     "--lama_ckpt", type=str, required=True,
    #     help="The path to the lama checkpoint.",
    # )
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    setup_args(parser)
    args = parser.parse_args(sys.argv[1:])
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    if args.coords_type == "click":
        latest_coords = get_clicked_point(args.input_img)
    elif args.coords_type == "box":
        # TODO
        latest_coords = get_box_point(args.input_img)
    elif args.coords_type == "brush":
        # TODO
        latest_coords = get_brush_point(args.input_img)
    elif args.coords_type == "key_in":
        # 作为API提供给前端服务
        latest_coords = args.point_coords

    latest_labels = args.point_labels
    
    # 选择MODEL segment anything
    if args.sam_ckpt != "./weights/mobile_sam.pt":
        model = FastSAM(args.sam_ckpt)
        # retina_masks 要求masks是高分辨率的，为了准确性
        results = model(args.input_img,device=DEVICE,retina_masks=True,imgsz=1024,conf=0.4,iou=0.9)
        prompt_process = FastSAMPrompt(args.input_img, results, device=DEVICE)
        # Point prompt
        # points default [[0,0]] [[x1,y1],[x2,y2]]
        # point_label default [0] [1,0] 0:background, 1:foreground
        ann = prompt_process.point_prompt(points=[latest_coords], pointlabel=latest_labels)
        prompt_process.plot(annotations=ann, output=args.output_dir)
        
        # plt.show()
        
    else:
        model = SAM(args.sam_ckpt)
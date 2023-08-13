import torch
import sys
import argparse
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

# from lama_inpaint import inpaint_img_with_lama
from utils import get_clicked_point, get_box_point, get_brush_point
from utils import sam_segment_object

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
        default="mobile_sam", choices=['FastSAM-s', 'FastSAM-x', 'mobile_sam'],
        help="The type of sam model to load. Default: 'mobile_sam"
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
    
    if args.coords_type == "click":
        point_coord_sets, point_label_sets = get_clicked_point(args.input_img)
    elif args.coords_type == "box":
        # TODO
        latest_coords = get_box_point(args.input_img)
    elif args.coords_type == "brush":
        # TODO
        latest_coords = get_brush_point(args.input_img)
    elif args.coords_type == "key_in":
        # 作为API提供给前端服务
        point_coord_sets, point_label_sets = args.point_coords, args.point_labels

    sam_segment_object(args.input_img, args.output_dir, args.sam_ckpt, point_coord_sets, point_label_sets)
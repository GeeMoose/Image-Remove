#!/bin/bash

python remove_anything.py \
    --input_img ./assets/lemons_output.jpg \
    --coords_type click \
    --point_coords 421 295 \
    --point_labels 1 \
    --dilate_kernel_size 15 \
    --output_dir ./results \
    --sam_model_type "FastSAM-s" \
    --sam_ckpt ./weights/FastSAM-s.pt
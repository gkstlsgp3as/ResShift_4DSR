#!/bin/bash

# updated: 08.14.2024
# Description: Apply inpainting and super-resolution onto Retina projection images
#
# Usage: bash sr_retina.sh
# -----------------------------------------------------------------------------

start_time=`date +%s`
# generate masks
python ./utils/generate_mask.py -i ../data/sample_retina/ -o ../data/sample_retina_mask/

# inpaint
python -m torch.distributed.launch --nproc_per_node 1 inference_resshift_4dsr.py -i ../data/sample_retina/ -o ../results/inpaint/ --mask_path ../data/sample_retina_mask/ --task inpaint_retina --scale 1

# resize 
python ./utils/resize.py -i ../results/inpaint -o ../results/inpaint_resize

# super-resolution
python -m torch.distributed.launch --nproc_per_node 1 inference_resshift_4dsr.py -i ../results/inpaint_resize -o ../results/sr/ --task retinasr --scale 4

# unnormalize 
python ./utils/unnormalize.py -i ../results/inpaint -o ../results/inpaint_unnorm

end_time=`date +%s`
echo execution time was `expr $end_time - $start_time` s.

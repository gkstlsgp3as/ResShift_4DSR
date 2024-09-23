#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2023-10-26 20:20:36

import argparse
from omegaconf import OmegaConf

from utils.util_common import get_obj_from_str
from utils.util_opts import str2bool

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
            "--save_dir",
            type=str,
            default="./save_dir",
            help="Folder to save the checkpoints and training log",
            )
    parser.add_argument(
            "--resume",
            type=str,
            const=True,
            default="",
            nargs="?",
            help="resume from the save_dir or checkpoint",
            )
    parser.add_argument(
            "--cfg_path",
            type=str,
            default="./configs/inpaint_lama256_retina.yaml",
            help="Configs of yaml file",
            )
    parser.add_argument('--local-rank', type=int, default=-1, help='DDP parameter, do not modify')
    
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = get_parser()

    configs = OmegaConf.load(args.cfg_path)

    # merge args to config
    for key in vars(args):
        if key in ['cfg_path', 'save_dir', 'resume', ]:
            configs[key] = getattr(args, key)

    trainer = get_obj_from_str(configs.trainer.target)(configs)
    trainer.train()

# python -m torch.distributed.launch --nproc_per_node 1 main.py # 디버깅 용
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 --nnodes=1 main.py --cfg_path configs/inpaint_lama256_retina.yaml --save_dir ../results/

## 환경세팅
# conda create -n resshift python=3.10
# conda activate resshift
# conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=12.1 -c pytorch -c nvidia
# pip install -r requirements.txt
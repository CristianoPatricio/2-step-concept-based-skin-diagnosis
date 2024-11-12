#!/bin/bash

#==============================================================================
# Title:        run_x_to_y.sh
# Description:  Script for replicating the results of Table 4
# Author:       Cristiano Patrício
# Date:         2024-11-11
# Version:      1.0
# Usage:        ./run_x_to_Y.sh {gpu_id}
#==============================================================================

GPU_ID="$1"

# CLIP
CUDA_VISIBLE_DEVICES=$GPU_ID python run_x_to_y.py --model=CLIP --dataset=Derm7pt
CUDA_VISIBLE_DEVICES=$GPU_ID python run_x_to_y.py --model=CLIP --dataset=HAM10000
CUDA_VISIBLE_DEVICES=$GPU_ID python run_x_to_y.py --model=CLIP --dataset=PH2 --split=0
CUDA_VISIBLE_DEVICES=$GPU_ID python run_x_to_y.py --model=CLIP --dataset=PH2 --split=1
CUDA_VISIBLE_DEVICES=$GPU_ID python run_x_to_y.py --model=CLIP --dataset=PH2 --split=2
CUDA_VISIBLE_DEVICES=$GPU_ID python run_x_to_y.py --model=CLIP --dataset=PH2 --split=3
CUDA_VISIBLE_DEVICES=$GPU_ID python run_x_to_y.py --model=CLIP --dataset=PH2 --split=4
python calculate_metrics.py --model=CLIP --task=PH2_eval --task_model=x_to_y --subdir=x_to_y

# BiomedCLIP
CUDA_VISIBLE_DEVICES=$GPU_ID python run_x_to_y.py --model=BiomedCLIP --dataset=Derm7pt
CUDA_VISIBLE_DEVICES=$GPU_ID python run_x_to_y.py --model=BiomedCLIP --dataset=HAM10000
CUDA_VISIBLE_DEVICES=$GPU_ID python run_x_to_y.py --model=BiomedCLIP --dataset=PH2 --split=0
CUDA_VISIBLE_DEVICES=$GPU_ID python run_x_to_y.py --model=BiomedCLIP --dataset=PH2 --split=1
CUDA_VISIBLE_DEVICES=$GPU_ID python run_x_to_y.py --model=BiomedCLIP --dataset=PH2 --split=2
CUDA_VISIBLE_DEVICES=$GPU_ID python run_x_to_y.py --model=BiomedCLIP --dataset=PH2 --split=3
CUDA_VISIBLE_DEVICES=$GPU_ID python run_x_to_y.py --model=BiomedCLIP --dataset=PH2 --split=4
python calculate_metrics.py --model=BiomedCLIP --task=PH2_eval --task_model=x_to_y --subdir=x_to_y

# MONET
CUDA_VISIBLE_DEVICES=$GPU_ID python run_x_to_y.py --model=MONET --dataset=Derm7pt
CUDA_VISIBLE_DEVICES=$GPU_ID python run_x_to_y.py --model=MONET --dataset=HAM10000
CUDA_VISIBLE_DEVICES=$GPU_ID python run_x_to_y.py --model=MONET --dataset=PH2 --split=0
CUDA_VISIBLE_DEVICES=$GPU_ID python run_x_to_y.py --model=MONET --dataset=PH2 --split=1
CUDA_VISIBLE_DEVICES=$GPU_ID python run_x_to_y.py --model=MONET --dataset=PH2 --split=2
CUDA_VISIBLE_DEVICES=$GPU_ID python run_x_to_y.py --model=MONET --dataset=PH2 --split=3
CUDA_VISIBLE_DEVICES=$GPU_ID python run_x_to_y.py --model=MONET --dataset=PH2 --split=4
python calculate_metrics.py --model=MONET --task=PH2_eval --task_model=x_to_y --subdir=x_to_y
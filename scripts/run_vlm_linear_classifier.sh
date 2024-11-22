#!/bin/bash

#==============================================================================
# Title:        run_vlm_linear_classifier.sh
# Description:  Script for replicating the results of VLM + Linear Classifier (Table 3)
# Author:       Cristiano Patrício
# Date:         2024-11-22
# Version:      1.0
# Usage:        ./run_vlm_linear_classifier.sh {gpu_id}
#==============================================================================

GPU_ID="$1"

# CLIP
CUDA_VISIBLE_DEVICES=$GPU_ID python run_vlm_linear_classifier.py --model=CLIP --dataset=Derm7pt
CUDA_VISIBLE_DEVICES=$GPU_ID python run_vlm_linear_classifier.py --model=CLIP --dataset=HAM10000
CUDA_VISIBLE_DEVICES=$GPU_ID python run_vlm_linear_classifier.py --model=CLIP --dataset=PH2 --split=0
CUDA_VISIBLE_DEVICES=$GPU_ID python run_vlm_linear_classifier.py --model=CLIP --dataset=PH2 --split=1
CUDA_VISIBLE_DEVICES=$GPU_ID python run_vlm_linear_classifier.py --model=CLIP --dataset=PH2 --split=2
CUDA_VISIBLE_DEVICES=$GPU_ID python run_vlm_linear_classifier.py --model=CLIP --dataset=PH2 --split=3
CUDA_VISIBLE_DEVICES=$GPU_ID python run_vlm_linear_classifier.py --model=CLIP --dataset=PH2 --split=4
python calculate_metrics.py --model=CLIP --task=PH2_eval --task_model=vlm_linear_classifier --subdir=vlm_linear_classifier

# BiomedCLIP
CUDA_VISIBLE_DEVICES=$GPU_ID python run_vlm_linear_classifier.py --model=BiomedCLIP --dataset=Derm7pt
CUDA_VISIBLE_DEVICES=$GPU_ID python run_vlm_linear_classifier.py --model=BiomedCLIP --dataset=HAM10000
CUDA_VISIBLE_DEVICES=$GPU_ID python run_vlm_linear_classifier.py --model=BiomedCLIP --dataset=PH2 --split=0
CUDA_VISIBLE_DEVICES=$GPU_ID python run_vlm_linear_classifier.py --model=BiomedCLIP --dataset=PH2 --split=1
CUDA_VISIBLE_DEVICES=$GPU_ID python run_vlm_linear_classifier.py --model=BiomedCLIP --dataset=PH2 --split=2
CUDA_VISIBLE_DEVICES=$GPU_ID python run_vlm_linear_classifier.py --model=BiomedCLIP --dataset=PH2 --split=3
CUDA_VISIBLE_DEVICES=$GPU_ID python run_vlm_linear_classifier.py --model=BiomedCLIP --dataset=PH2 --split=4
python calculate_metrics.py --model=BiomedCLIP --task=PH2_eval --task_model=vlm_linear_classifier --subdir=vlm_linear_classifier

# MONET
CUDA_VISIBLE_DEVICES=$GPU_ID python run_vlm_linear_classifier.py --model=MONET --dataset=Derm7pt
CUDA_VISIBLE_DEVICES=$GPU_ID python run_vlm_linear_classifier.py --model=MONET --dataset=HAM10000
CUDA_VISIBLE_DEVICES=$GPU_ID python run_vlm_linear_classifier.py --model=MONET --dataset=PH2 --split=0
CUDA_VISIBLE_DEVICES=$GPU_ID python run_vlm_linear_classifier.py --model=MONET --dataset=PH2 --split=1
CUDA_VISIBLE_DEVICES=$GPU_ID python run_vlm_linear_classifier.py --model=MONET --dataset=PH2 --split=2
CUDA_VISIBLE_DEVICES=$GPU_ID python run_vlm_linear_classifier.py --model=MONET --dataset=PH2 --split=3
CUDA_VISIBLE_DEVICES=$GPU_ID python run_vlm_linear_classifier.py --model=MONET --dataset=PH2 --split=4
python calculate_metrics.py --model=MONET --task=PH2_eval --task_model=vlm_linear_classifier --subdir=vlm_linear_classifier

# ExpLICD
CUDA_VISIBLE_DEVICES=$GPU_ID python run_vlm_linear_classifier.py --model=ExpLICD --dataset=Derm7pt
CUDA_VISIBLE_DEVICES=$GPU_ID python run_vlm_linear_classifier.py --model=ExpLICD --dataset=HAM10000
CUDA_VISIBLE_DEVICES=$GPU_ID python run_vlm_linear_classifier.py --model=ExpLICD --dataset=PH2 --split=0
CUDA_VISIBLE_DEVICES=$GPU_ID python run_vlm_linear_classifier.py --model=ExpLICD --dataset=PH2 --split=1
CUDA_VISIBLE_DEVICES=$GPU_ID python run_vlm_linear_classifier.py --model=ExpLICD --dataset=PH2 --split=2
CUDA_VISIBLE_DEVICES=$GPU_ID python run_vlm_linear_classifier.py --model=ExpLICD --dataset=PH2 --split=3
CUDA_VISIBLE_DEVICES=$GPU_ID python run_vlm_linear_classifier.py --model=ExpLICD --dataset=PH2 --split=4
python calculate_metrics.py --model=ExpLICD --task=PH2_eval --task_model=vlm_linear_classifier --subdir=vlm_linear_classifier

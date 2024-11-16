#!/bin/bash

#==============================================================================
# Title:        run_x_to_c_to_y.sh
# Description:  Script for replicating the results of Table 3
# Author:       Cristiano Patrício
# Date:         2024-11-11
# Version:      1.0
# Usage:        ./run_x_to_c_to_y.sh {gpu_id}
#==============================================================================

GPU_ID="$1"

###################################
# Few-Shot (1-shot) | ExpLICD
###################################

# NOTE: You could replace --n_demos by 2,4,8

# Mistral
CUDA_VISIBLE_DEVICES=$GPU_ID python run_x_to_c_to_y.py --dataset=Derm7pt --concept_extractor=Explicd --llm=Mistral --use_demos --n_demos=1
CUDA_VISIBLE_DEVICES=$GPU_ID python run_x_to_c_to_y.py --dataset=HAM10000 --concept_extractor=Explicd --llm=Mistral --use_demos --n_demos=1
CUDA_VISIBLE_DEVICES=$GPU_ID python run_x_to_c_to_y.py --dataset=PH2 --concept_extractor=Explicd --llm=Mistral --use_demos --n_demos=1 --split=0
CUDA_VISIBLE_DEVICES=$GPU_ID python run_x_to_c_to_y.py --dataset=PH2 --concept_extractor=Explicd --llm=Mistral --use_demos --n_demos=1 --split=1
CUDA_VISIBLE_DEVICES=$GPU_ID python run_x_to_c_to_y.py --dataset=PH2 --concept_extractor=Explicd --llm=Mistral --use_demos --n_demos=1 --split=2
CUDA_VISIBLE_DEVICES=$GPU_ID python run_x_to_c_to_y.py --dataset=PH2 --concept_extractor=Explicd --llm=Mistral --use_demos --n_demos=1 --split=3
CUDA_VISIBLE_DEVICES=$GPU_ID python run_x_to_c_to_y.py --dataset=PH2 --concept_extractor=Explicd --llm=Mistral --use_demos --n_demos=1 --split=4
python calculate_metrics.py --model=Mistral --task=PH2_eval --task_model=gt_concepts_False_raw_values_False_model_extractor_Explicd_n_demos_1 --subdir=x_to_c_to_y

# MMed
CUDA_VISIBLE_DEVICES=$GPU_ID python run_x_to_c_to_y.py --dataset=Derm7pt --concept_extractor=Explicd --llm=MMed --ckpt=Henrychur/MMed-Llama-3-8B-EnIns --use_demos --n_demos=1
CUDA_VISIBLE_DEVICES=$GPU_ID python run_x_to_c_to_y.py --dataset=HAM10000 --concept_extractor=Explicd --llm=MMed --ckpt=Henrychur/MMed-Llama-3-8B-EnIns --use_demos --n_demos=1
CUDA_VISIBLE_DEVICES=$GPU_ID python run_x_to_c_to_y.py --dataset=PH2 --concept_extractor=Explicd --llm=MMed --ckpt=Henrychur/MMed-Llama-3-8B-EnIns --use_demos --n_demos=1 --split=0
CUDA_VISIBLE_DEVICES=$GPU_ID python run_x_to_c_to_y.py --dataset=PH2 --concept_extractor=Explicd --llm=MMed --ckpt=Henrychur/MMed-Llama-3-8B-EnIns --use_demos --n_demos=1 --split=1
CUDA_VISIBLE_DEVICES=$GPU_ID python run_x_to_c_to_y.py --dataset=PH2 --concept_extractor=Explicd --llm=MMed --ckpt=Henrychur/MMed-Llama-3-8B-EnIns --use_demos --n_demos=1 --split=2
CUDA_VISIBLE_DEVICES=$GPU_ID python run_x_to_c_to_y.py --dataset=PH2 --concept_extractor=Explicd --llm=MMed --ckpt=Henrychur/MMed-Llama-3-8B-EnIns --use_demos --n_demos=1 --split=3
CUDA_VISIBLE_DEVICES=$GPU_ID python run_x_to_c_to_y.py --dataset=PH2 --concept_extractor=Explicd --llm=MMed --ckpt=Henrychur/MMed-Llama-3-8B-EnIns --use_demos --n_demos=1 --split=4
python calculate_metrics.py --model=MMed --task=PH2_eval --task_model=gt_concepts_False_raw_values_False_model_extractor_Explicd_n_demos_1 --subdir=x_to_c_to_y

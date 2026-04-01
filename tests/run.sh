#!/bin/bash
set -euo pipefail

export HF_HOME="/data/jiayi"
export MODEL_NAME='DeepSeek-R1-Distill-Qwen-1.5B'
export MODEL_PATH='deepseek-ai/'$MODEL_NAME
export CUDA_VISIBLE_DEVICES=7

# STATIC_STEER_ENABLE=0 python test_steer.py

# STATIC_STEER_ENABLE=1 \
# STATIC_STEER_LAYER=20 \
# STATIC_STEER_MATCH_TOKEN_IDS=271 \
# STATIC_STEER_PATH="/data/jiayi/SEAL/results/MATH_train/DeepSeek-R1-Distill-Qwen-1.5B/baseline_10000/vector_500_500/layer_20_transition_reflection_steervec.pt" \
# STATIC_STEER_SCALE=-1.0 \
# python test_steer.py

for RANK in 65 70 75 
do
STATIC_STEER_ENABLE=1 \
STATIC_STEER_LAYER=20 \
STATIC_STEER_MATCH_TOKEN_IDS=271 \
STATIC_STEER_PATH="/home/jiayi/TensorRouter/TensorRouter/vector_500_500/DeepSeek-R1-Distill-Qwen-1.5B/layer_21_highrank_${RANK}_transition_reflection_steervec.pt" \
STATIC_STEER_SCALE=1.0 \
python test_steer.py
done 

# STATIC_STEER_ENABLE=0 python test_steer_serve.py


# for STATIC_STEER_SCALE in 1.0; do
# 	STATIC_STEER_ENABLE=1 \
# 	STATIC_STEER_LAYER=20 \
#     STATIC_STEER_MATCH_TOKEN_IDS=271 \
# 	STATIC_STEER_PATH="/home/jiayi/TensorRouter/TensorRouter/vector_500_500/$MODEL_NAME/layer_21_highrank_60_transition_reflection_steervec.pt" \
# 	STATIC_STEER_SCALE="${STATIC_STEER_SCALE}" \
# 	python test_steer_serve.py
# done

# for STATIC_STEER_SCALE in -1.0; do
# 	STATIC_STEER_ENABLE=1 \
# 	STATIC_STEER_LAYER=20 \
#     STATIC_STEER_MATCH_TOKEN_IDS=271 \
# 	STATIC_STEER_PATH="/home/jiayi/TensorRouter/TensorRouter/vector_500_500/$MODEL_NAME/layer_20_transition_reflection_steervec.pt" \
# 	STATIC_STEER_SCALE="${STATIC_STEER_SCALE}" \
# 	python test_steer_serve.py
# done
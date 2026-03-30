#!/bin/bash
set -euo pipefail

export HF_HOME="/data/jiayi"
export MODEL_NAME='Qwen3-4B'
export MODEL_PATH='Qwen/'$MODEL_NAME
export CUDA_VISIBLE_DEVICES=1

STATIC_STEER_ENABLE=0 python test_steer.py

STATIC_STEER_ENABLE=1 \
STATIC_STEER_LAYER=28 \
STATIC_STEER_MATCH_TOKEN_IDS=271 \
STATIC_STEER_PATH="/data/jiayi/SEAL/results/MATH_train/Qwen3-4B/baseline_10000/vector_500_500/layer_28_transition_reflection_steervec.pt" \
STATIC_STEER_SCALE=-1.0 \
python test_steer.py

for RANK in 50 60 70; do
    export RANK
    STATIC_STEER_ENABLE=1 \
    STATIC_STEER_LAYER=28 \
    STATIC_STEER_MATCH_TOKEN_IDS=271 \
    STATIC_STEER_PATH="/data/jiayi/SEAL/results/MATH_train/Qwen3-4B/baseline_10000/vector_500_500/layer_28_rank_${RANK}_transition_reflection_steervec.pt" \
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
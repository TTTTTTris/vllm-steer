#!/bin/bash
set -euo pipefail

export MODEL_NAME='DeepSeek-R1-Distill-Qwen-1.5B'
export MODEL_PATH='deepseek-ai/'$MODEL_NAME
STATIC_STEER_ENABLE=0 python test_steer.py

for STATIC_STEER_SCALE in 0.5 1.0; do
	STATIC_STEER_ENABLE=1 \
	STATIC_STEER_LAYER=20 \
	STATIC_STEER_PATH="/home/jiayi/TensorRouter/TensorRouter/vector_500_500/$MODEL_NAME/layer_21_highrank_60_transition_reflection_steervec.pt" \
	STATIC_STEER_SCALE="${STATIC_STEER_SCALE}" \
	python test_steer.py
done

for STATIC_STEER_SCALE in -0.5 -1.0; do
	STATIC_STEER_ENABLE=1 \
	STATIC_STEER_LAYER=20 \
	STATIC_STEER_PATH="/home/jiayi/TensorRouter/TensorRouter/vector_500_500/$MODEL_NAME/layer_20_transition_reflection_steervec.pt" \
	STATIC_STEER_SCALE="${STATIC_STEER_SCALE}" \
	python test_steer.py
done
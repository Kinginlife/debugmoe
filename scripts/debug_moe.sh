#!/bin/bash
# Debug script - use GT masks to isolate MoE issues

SCRIPT_DIR=$(cd "$(dirname "$0")" &> /dev/null && pwd)
cd "$SCRIPT_DIR/.."

export DETECTRON2_DATASETS=/data1/lsh/VITA-main/datasets
export CUDA_VISIBLE_DEVICES=4,5,6,7

NGPUS=4
CFG_FILE="configs/youtubevis_2019/vita_R50_bs8_moe_debug.yaml"
OUTPUT_BASE="output/ytvis_2019_moe_debug"
EXP_NAME="VITA_MoE_Debug"

STEP_ARGS="CONT.BASE_CLS 20 CONT.INC_CLS 2 CONT.MODE overlap SEED 42"
METH_ARGS="MODEL.MASK_FORMER.TEST.MASK_BG False MODEL.MASK_FORMER.PER_PIXEL False MODEL.MASK_FORMER.FOCAL True"

BASE_QUERIES=100
ITER_BASE=10000
ITER_INC=2500

WEIGHT_ARGS="MODEL.MASK_FORMER.NUM_OBJECT_QUERIES ${BASE_QUERIES} \
             MODEL.VITA.NUM_OBJECT_QUERIES ${BASE_QUERIES} \
             MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME VitaMultiScaleMaskedTransformerDecoder \
             MODEL.MASK_FORMER.USE_MOE True"

COMM_ARGS="${STEP_ARGS} ${WEIGHT_ARGS} ${METH_ARGS}"

echo "=========================================="
echo "DEBUG MODE: Using GT masks"
echo "Only training: class_embed + new Expert"
echo "=========================================="

# Task 0
OUT_DIR_0="${OUTPUT_BASE}/step0"
echo ">>> Training Task 0 (Base)"
python train_incremental_moe.py --num-gpus ${NGPUS} \
    --dist-url tcp://127.0.0.1:50164 \
    --config-file ${CFG_FILE} \
    OUTPUT_DIR ${OUT_DIR_0} \
    CONT.TASK 0 \
    CONT.WEIGHTS vita_r50_coco.pth \
    SOLVER.MAX_ITER ${ITER_BASE} \
    ${COMM_ARGS}

# Task 1
OUT_DIR_1="${OUTPUT_BASE}/step1"
PRETRAINED_PATH="${OUT_DIR_0}/model_final.pth"
echo ">>> Training Task 1 (DEBUG MODE - GT Masks)"
python train_incremental_moe.py --num-gpus ${NGPUS} \
    --dist-url tcp://127.0.0.1:50164 \
    --config-file ${CFG_FILE} \
    OUTPUT_DIR ${OUT_DIR_1} \
    CONT.WEIGHTS ${PRETRAINED_PATH} \
    CONT.TASK 1 \
    SOLVER.MAX_ITER ${ITER_INC} \
    ${COMM_ARGS}

echo ">>> Debug training completed!"

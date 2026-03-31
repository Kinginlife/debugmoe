#!/bin/bash
SCRIPT_DIR=$(cd "$(dirname "$0")" &> /dev/null && pwd)
cd "$SCRIPT_DIR/.."

export DETECTRON2_DATASETS=/data1/lsh/VITA-main/datasets
export CUDA_VISIBLE_DEVICES=4,5,6,7

NGPUS=4
CFG_FILE="configs/youtubevis_2019/vita_R50_bs8.yaml"
OUTPUT_BASE="output/ytvis_2019_moe"
EXP_NAME="VITA_MoE_20_2"

STEP_ARGS="CONT.BASE_CLS 20 CONT.INC_CLS 2 CONT.MODE overlap SEED 42"
METH_ARGS="MODEL.MASK_FORMER.TEST.MASK_BG False MODEL.MASK_FORMER.PER_PIXEL False MODEL.MASK_FORMER.FOCAL True"

BASE_QUERIES=100
ITER_BASE=10000



WEIGHT_ARGS="MODEL.MASK_FORMER.NUM_OBJECT_QUERIES ${BASE_QUERIES} \
             MODEL.VITA.NUM_OBJECT_QUERIES ${BASE_QUERIES} \
             MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME VitaMultiScaleMaskedTransformerDecoder"

COMM_ARGS="${STEP_ARGS} ${WEIGHT_ARGS} ${METH_ARGS}"

# ==========================================
# Task 0 (Base)
# ==========================================
OUT_DIR_0="${OUTPUT_BASE}/step0"
INC_ARGS_0="CONT.TASK 0 \
            TEST.EVAL_PERIOD 2000 \
            SOLVER.CHECKPOINT_PERIOD 5000 \
            CONT.WEIGHTS vita_r50_coco.pth \
            SOLVER.MAX_ITER ${ITER_BASE} \
            "

echo ">>> Training Task 0 (Base)"
python train_net_vita.py --num-gpus ${NGPUS} \
    --dist-url tcp://127.0.0.1:50164 \
    --config-file ${CFG_FILE} \
    OUTPUT_DIR ${OUT_DIR_0} \
    ${COMM_ARGS} ${INC_ARGS_0}


# ==========================================
# Task 1 (First Incremental)
# ==========================================
ITER_INC=2500
BASE_QUERIES_INC=100

FREEZE_ARGS="CONT.FREEZE_BACKBONE True"

WEIGHT_ARGS_INC="MODEL.MASK_FORMER.NUM_OBJECT_QUERIES ${BASE_QUERIES_INC} \
                 MODEL.VITA.NUM_OBJECT_QUERIES ${BASE_QUERIES_INC} \
                 MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME VitaMultiScaleMaskedTransformerDecoder"


COMM_ARGS_INC="${STEP_ARGS} ${WEIGHT_ARGS_INC} "

OUT_DIR_1="${OUTPUT_BASE}/step1"

PRETRAINED_PATH="${OUT_DIR_0}/model_final.pth" 

echo ">>> Training Task 1"
python train_net_vita.py --num-gpus ${NGPUS} \
    --config-file ${CFG_FILE} \
    OUTPUT_DIR ${OUT_DIR_1} \
    CONT.WEIGHTS ${PRETRAINED_PATH} \
    TEST.EVAL_PERIOD 100 \
    CONT.TASK 1 \
    SOLVER.MAX_ITER ${ITER_INC} \
    ${COMM_ARGS_INC} ${FREEZE_ARGS}



# ==========================================
# Task 2..10 (Rest Incremental Steps)
# ==========================================
for t in {2..10}; do
    CURR_OUT="${OUTPUT_BASE}/step${t}"
    PREV_WEIGHTS="${OUTPUT_BASE}/step${t-1}/model_final.pth"

    echo ">>> Training Task ${t}"
    python train_net_vita.py --num-gpus ${NGPUS} \
        --config-file ${CFG_FILE} \
        OUTPUT_DIR ${CURR_OUT} \
        CONT.WEIGHTS ${PREV_WEIGHTS} \
        CONT.TASK ${t} \
        SOLVER.MAX_ITER ${ITER_INC} \
        ${COMM_ARGS_INC} ${FREEZE_ARGS}
done
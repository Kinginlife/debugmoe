#!/bin/bash
SCRIPT_DIR=$(cd "$(dirname "$0")" &> /dev/null && pwd)
cd "$SCRIPT_DIR/.."

export DETECTRON2_DATASETS=/data1/lsh/VITA_continue/datasets
export CUDA_VISIBLE_DEVICES=4,5,6,7

NGPUS=4
CFG_FILE="configs/youtubevis_2019/vita_R50_bs8_moe.yaml"
OUTPUT_BASE="output_base4ep/ytvis_2019_moe"
#OUTPUT_BASE="output3/ytvis_2019_moe"
EXP_NAME="VITA_MoE_20_2"

STEP_ARGS="CONT.BASE_CLS 20 CONT.INC_CLS 2 CONT.MODE overlap SEED 42"

BASE_QUERIES=100
ITER_BASE=20000
# Configure base task expert count here (task0 experts)
BASE_EXPERTS=4


WEIGHT_ARGS="MODEL.MASK_FORMER.NUM_OBJECT_QUERIES ${BASE_QUERIES} \
             MODEL.VITA.NUM_OBJECT_QUERIES ${BASE_QUERIES} \
             MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME VitaMultiScaleMaskedTransformerDecoder \
             MODEL.MASK_FORMER.USE_MOE True "

COMM_ARGS="${STEP_ARGS} ${WEIGHT_ARGS}"

# ==========================================
# Task 0 (Base)
# ==========================================
OUT_DIR_0="${OUTPUT_BASE}/step0"
INC_ARGS_0="CONT.TASK 0 \
            TEST.EVAL_PERIOD 5000 \
            SOLVER.CHECKPOINT_PERIOD 10000 \
            CONT.WEIGHTS vita_r50_coco.pth \
            CONT.BASE_EXPERTS ${BASE_EXPERTS} \
            SOLVER.MAX_ITER ${ITER_BASE}"

echo ">>> Training Task 0 (Base) with MoE"
python train_incremental_moe.py --num-gpus ${NGPUS} \
    --dist-url tcp://127.0.0.1:50164 \
    --config-file ${CFG_FILE} \
    OUTPUT_DIR ${OUT_DIR_0} \
    ${COMM_ARGS} ${INC_ARGS_0}


# ==========================================
# Task 1 (First Incremental)
# ==========================================
ITER_INC=2500
BASE_QUERIES_INC=100

WEIGHT_ARGS_INC="MODEL.MASK_FORMER.NUM_OBJECT_QUERIES ${BASE_QUERIES_INC} \
                 MODEL.VITA.NUM_OBJECT_QUERIES ${BASE_QUERIES_INC} \
                 MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME VitaMultiScaleMaskedTransformerDecoder \
                 MODEL.MASK_FORMER.USE_MOE True \
                 CONT.BASE_EXPERTS ${BASE_EXPERTS}"

COMM_ARGS_INC="${STEP_ARGS} ${WEIGHT_ARGS_INC}"

OUT_DIR_1="${OUTPUT_BASE}/step1"
PRETRAINED_PATH="${OUT_DIR_0}/model_final.pth"

echo ">>> Training Task 1 with MoE (adding Expert 1, freezing shared components)"
python train_incremental_moe.py --num-gpus ${NGPUS} \
    --dist-url tcp://127.0.0.1:50164 \
    --config-file ${CFG_FILE} \
    OUTPUT_DIR ${OUT_DIR_1} \
    CONT.WEIGHTS ${PRETRAINED_PATH} \
    TEST.EVAL_PERIOD 1000 \
    CONT.TASK 1 \
    SOLVER.MAX_ITER ${ITER_INC} \
    ${COMM_ARGS_INC}



# ==========================================
# Task 2..10 (Rest Incremental Steps)
# ==========================================
for t in {2..10}; do
    CURR_OUT="${OUTPUT_BASE}/step${t}"
    PREV_STEP=$((t-1))
    PREV_WEIGHTS="${OUTPUT_BASE}/step${PREV_STEP}/model_task${PREV_STEP}_router_merged.pth"

    echo ">>> Training Task ${t} with MoE (adding Expert ${t}, freezing shared components)"
    python train_incremental_moe.py --num-gpus ${NGPUS} \
        --dist-url tcp://127.0.0.1:50164 \
        --config-file ${CFG_FILE} \
        OUTPUT_DIR ${CURR_OUT} \
        CONT.WEIGHTS ${PREV_WEIGHTS} \
        TEST.EVAL_PERIOD 1000 \
        CONT.TASK ${t} \
        SOLVER.MAX_ITER ${ITER_INC} \
        ${COMM_ARGS_INC}
done

echo ">>> All tasks completed!"
echo ">>> Results saved in: ${OUTPUT_BASE}"
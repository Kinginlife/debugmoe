#!/bin/bash
SCRIPT_DIR=$(cd "$(dirname "$0")" &> /dev/null && pwd)
cd "$SCRIPT_DIR/.."
# 设定显卡
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 运行刚刚写的生成脚本
python /data1/lsh/VITA_continue/step0svd.py \
    --num-gpus 4 \
    --dist-url tcp://127.0.0.1:20165 \
    --config-file configs/youtubevis_2019/vita_R50_bs8_moe.yaml \
    MODEL.WEIGHTS /data1/lsh/VITA_continue/output_base4epiter1000/ytvis_2019_moe/step0/model_final.pth \
    OUTPUT_DIR /data1/lsh/VITA_continue/output_base4epiter1000/ytvis_2019_moe/step0 \
    CONT.BASE_EXPERTS 2 \
    CONT.SVD_THRESHOLD 0.7
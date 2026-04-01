# MoE-based Incremental Learning for VITA

## 概述

本实现在 VITA 的 Frame Decoder 最后一层添加了 Mixture of Experts (MoE)，实现增量学习。

## 架构设计

```
Frame Decoder (9层)
├─ Layer 0-7: 标准 FFN (所有任务共享)
└─ Layer 8: MoE FFN (任务特定)
    ├─ Expert 0: Task 0
    ├─ Expert 1: Task 1
    └─ Expert N: Task N
```

## 训练策略

### Task 0 (Base Task)
- **所有组件**: 可训练
- Expert 0: 可训练
- Router: 可训练

### Task 1+
- **冻结组件**:
  - Backbone (特征提取器)
  - Pixel Decoder
  - Frame Decoder 的 Self-Attention 层
  - Frame Decoder 的 Cross-Attention 层
  - Frame Decoder 的前 N-1 层 FFN
  - 旧的 Experts (0~N-1)
  - Router

- **可训练组件**:
  - Query embeddings (query_feat, query_embed) - 适应新任务查询
  - Prediction heads (class_embed, mask_embed) - 学习新类别
  - 新的 Expert N (仅在最后一层 MoE FFN 中)

## 使用方法

### 1. Task 0 训练

```bash
python train_incremental_moe.py \
    --config-file configs/youtubevis_2019/vita_R50_bs8_moe.yaml \
    --num-gpus 8 \
    CONT.TASK 0 \
    OUTPUT_DIR output/task0
```

### 2. Task 1 训练

```bash
python train_incremental_moe.py \
    --config-file configs/youtubevis_2019/vita_R50_bs8_moe.yaml \
    --num-gpus 8 \
    CONT.TASK 1 \
    CONT.WEIGHTS output/task0/model_final.pth \
    OUTPUT_DIR output/task1
```

### 3. Task N 训练

```bash
python train_incremental_moe.py \
    --config-file configs/youtubevis_2019/vita_R50_bs8_moe.yaml \
    --num-gpus 8 \
    CONT.TASK N \
    CONT.WEIGHTS output/task{N-1}/model_final.pth \
    OUTPUT_DIR output/taskN
```

## 配置参数

在配置文件中设置：

```yaml
MODEL:
  MASK_FORMER:
    USE_MOE: True  # 启用 MoE

CONT:
  TASK: 0  # 当前任务 ID
  WEIGHTS: null  # Task 1+ 需要指定上一个任务的权重路径
```

## 文件说明

- `moe_layer.py`: MoE FFN Layer 实现
- `vita_mask2former_transformer_decoder.py`: 修改后的 Frame Decoder
- `train_incremental_moe.py`: 增量学习训练脚本
- `configs/youtubevis_2019/vita_R50_bs8_moe.yaml`: MoE 配置示例

## 关键特性

1. **参数隔离**: 每个任务有独立的 Expert
2. **避免遗忘**: 旧 Expert 被冻结
3. **知识复用**: Router 学习何时使用哪个 Expert
4. **最小修改**: 只修改最后一层 FFN

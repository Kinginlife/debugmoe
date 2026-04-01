# Frame Decoder MoE 增量学习实现方案

## 已完成的修改

### 1. 核心文件

#### ✅ `vita/modeling/transformer_decoder/moe_layer.py`
- 实现了 `MoEFFNLayer` 类
- 支持多个 Expert 的加权组合
- 提供 `add_new_expert()` 方法添加新任务的 Expert
- 提供 `freeze_old_experts()` 方法冻结旧任务参数

#### ✅ `vita/modeling/transformer_decoder/vita_mask2former_transformer_decoder.py`
- 导入 `MoEFFNLayer`
- 修改 `__init__` 添加 MoE 参数
- 在最后一层 FFN 使用 MoE（当 `use_moe=True` 时）
- 修改 `from_config` 读取 MoE 配置
- 添加 `add_new_expert_for_task()` 方法支持任务切换

#### ✅ `mask2former/config.py`
- 添加 `cfg.MODEL.MASK_FORMER.USE_MOE` 配置项

### 2. 配置和脚本

#### ✅ `configs/youtubevis_2019/vita_R50_bs8_moe.yaml`
- MoE 配置示例

#### ✅ `train_incremental_moe.py`
- 增量学习训练脚本
- 自动加载上一任务模型
- 自动添加新 Expert 并冻结旧参数

#### ✅ `test_moe.py`
- MoE 单元测试脚本

#### ✅ `MOE_README.md`
- 使用说明文档

## 实现原理

### 架构

```
Frame Decoder (VitaMultiScaleMaskedTransformerDecoder)
├─ Layer 0-7: 标准 FFN (共享)
└─ Layer 8: MoE FFN
    ├─ Router: Linear(d_model, num_experts)
    └─ Experts: [Expert_0, Expert_1, ..., Expert_N]
```

### MoE 前向传播

```python
# 1. Router 计算权重
router_logits = router(x)  # [seq, batch, num_experts]
router_weights = softmax(router_logits, dim=-1)

# 2. 每个 Expert 独立计算
for i, expert in enumerate(experts):
    expert_out = expert(x)
    weighted_out = expert_out * router_weights[..., i:i+1]
    outputs.append(weighted_out)

# 3. 加权求和
output = sum(outputs)
```

### 增量学习流程

**Task 0**:
- 初始化 1 个 Expert
- 所有组件都可训练

**Task 1**:
- 加载 Task 0 模型
- 添加 Expert 1
- **冻结**:
  - Backbone
  - Pixel Decoder
  - Frame Decoder 所有层（除了最后一层的新 Expert）
  - Expert 0 和 Router
- **可训练**: Expert 1

**Task N**:
- 加载 Task N-1 模型
- 添加 Expert N
- **冻结**: 所有共享组件 + Expert 0~N-1 + Router
- **可训练**: Expert N

### 参数冻结详情

增量任务时冻结的组件：
1. Backbone (ResNet/Swin)
2. Pixel Decoder (MSDeformAttnPixelDecoder)
3. Frame Decoder:
   - Self-Attention 层 (所有层)
   - Cross-Attention 层 (所有层)
   - FFN 层 (前 N-1 层)
   - Input projections
   - Level embeddings
   - Decoder norm
4. MoE 组件:
   - 旧 Experts (0~N-1)
   - Router

增量任务时保持可训练的组件：
1. Query embeddings (query_feat, query_embed) - 适应新任务
2. Prediction heads:
   - class_embed - 学习新类别分类
   - mask_embed - 学习新类别 mask 特征
3. 新 Expert N - 学习新任务特定的帧特征

## 使用步骤

### 1. 运行测试
```bash
cd VITA_continue
python test_moe.py
```

### 2. Task 0 训练
```bash
python train_incremental_moe.py \
    --config-file configs/youtubevis_2019/vita_R50_bs8_moe.yaml \
    --num-gpus 8 \
    CONT.TASK 0 \
    OUTPUT_DIR output/task0
```

### 3. Task 1 训练
```bash
python train_incremental_moe.py \
    --config-file configs/youtubevis_2019/vita_R50_bs8_moe.yaml \
    --num-gpus 8 \
    CONT.TASK 1 \
    CONT.WEIGHTS output/task0/model_final.pth \
    OUTPUT_DIR output/task1
```

## 关键代码位置

| 功能 | 文件 | 行数 |
|------|------|------|
| MoE Layer 定义 | moe_layer.py | 全文 |
| Frame Decoder MoE 集成 | vita_mask2former_transformer_decoder.py | 13, 233-251, 283-309, 367-380 |
| 配置项 | mask2former/config.py | 117 |
| 训练脚本 | train_incremental_moe.py | 全文 |

## 下一步工作

如果需要在 VITA Decoder 也添加 MoE：
1. 修改 `vita/modeling/transformer_decoder/vita.py`
2. 在第 301-308 行应用相同的 MoE 逻辑
3. 在 `vita_model.py` 中添加任务切换接口


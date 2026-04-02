# MoE Debug 方案

## 问题现象

Task 1 训练后：
- Task 0 AP 迅速下降
- Task 1 AP 训练较好

## 可能原因

1. 参数冻结不完全
2. Mask 预测部分影响
3. Router 未真正冻结

## Debug 策略

### 方法：隔离变量

**冻结 mask 预测部分**，只训练：
- Frame Decoder: class_embed + 新 Expert
- VITA module: class_embed

这样可以：
1. 消除 mask 预测的影响
2. 专注于分类和 MoE 的问题
3. 验证冻结策略是否有效

## 使用方法

### 1. 运行 Debug 训练

```bash
cd /data1/lsh/VITA_continue
bash scripts/debug_moe.sh
```

### 2. 检查输出

查看 `output/ytvis_2019_moe_debug/step1/canshu.txt`

确认：
- ✅ mask_embed 被冻结
- ✅ 只有 class_embed 和新 Expert 可训练
- ✅ 旧 Expert 和 Router 被冻结

### 3. 观察结果

如果 Task 0 性能仍然下降：
- 说明冻结策略有问题
- 需要检查 Router 或旧 Expert 是否真的冻结

如果 Task 0 性能保持：
- 说明问题在 mask 预测部分
- 需要进一步调试 mask_embed 的训练策略

## 文件说明

- `train_debug_moe.py`: Debug 训练脚本（冻结 mask 预测）
- `scripts/debug_moe.sh`: Debug 训练启动脚本
- `train_incremental_moe.py`: 正常训练脚本

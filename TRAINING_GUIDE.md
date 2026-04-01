# 训练脚本使用说明

## 脚本位置
`scripts/ytb2019_20_2.sh`

## 主要修改

1. **配置文件**: 使用 `vita_R50_bs8_moe.yaml` (启用 MoE)
2. **训练脚本**: 使用 `train_incremental_moe.py` (自动冻结和添加 Expert)
3. **MoE 启用**: `MODEL.MASK_FORMER.USE_MOE True`

## 训练流程

### Task 0 (Base)
- 训练 Expert 0
- 所有参数可训练
- 迭代次数: 10000

### Task 1-10 (Incremental)
- 自动加载上一任务模型
- 自动添加新 Expert
- 自动冻结共享组件
- 迭代次数: 2500

## 运行命令

```bash
cd /Users/lsh21/Downloads/debugmoe/VITA_continue
bash scripts/ytb2019_20_2.sh
```

## 输出目录

```
output/ytvis_2019_moe/
├── step0/  # Task 0
├── step1/  # Task 1
├── step2/  # Task 2
...
└── step10/ # Task 10
```

## 关键参数

- `CONT.BASE_CLS 20`: 基础任务 20 个类别
- `CONT.INC_CLS 2`: 每个增量任务 2 个新类别
- `CONT.MODE overlap`: 重叠模式
- `NGPUS 4`: 使用 4 个 GPU
- `CUDA_VISIBLE_DEVICES=4,5,6,7`: GPU 编号

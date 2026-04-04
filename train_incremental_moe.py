"""
Incremental Learning Training Script with MoE

Usage:
    # Task 0 (Base task)
    python train_incremental_moe.py --config-file configs/youtubevis_2019/vita_R50_bs8_moe.yaml \
        --num-gpus 8 CONT.TASK 0

    # Task 1 (Load Task 0 model and add new expert)
    python train_incremental_moe.py --config-file configs/youtubevis_2019/vita_R50_bs8_moe.yaml \
        --num-gpus 8 CONT.TASK 1 CONT.WEIGHTS output/task0/model_final.pth
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from train_net_vita import Trainer, default_argument_parser, default_setup, launch
from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.projects.deeplab import add_deeplab_config
from mask2former import add_maskformer2_config
from vita import add_vita_config
from continual import add_continual_config
import torch
import torch.distributed as dist
import detectron2.utils.comm as comm

class SVDManager:
    def __init__(self, feature_dim=256):
        self.cov_frame = torch.zeros(feature_dim, feature_dim).cuda()
        self.cov_clip = torch.zeros(feature_dim, feature_dim).cuda()

    def hook_frame(self, module, input, output):
        # 收集 Frame 级特征
        x = input[0].detach().reshape(-1, input[0].shape[-1])
        self.cov_frame += torch.matmul(x.T, x)

    def hook_clip(self, module, input, output):
        # 收集 Clip (VITA) 级特征
        x = input[0].detach().reshape(-1, input[0].shape[-1])
        self.cov_clip += torch.matmul(x.T, x)

    def sync_covariance(self):
        """DDP 多卡环境下，将所有 GPU 的协方差矩阵同步求和"""
        if comm.get_world_size() > 1:
            dist.all_reduce(self.cov_frame, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.cov_clip, op=dist.ReduceOp.SUM)

    @staticmethod
    def update_basis(old_U, cov_matrix, threshold=0.7):
        # 1. 投影到老任务零空间
        if old_U is not None:
            I = torch.eye(cov_matrix.shape[0], device=cov_matrix.device)
            P_old = I - torch.matmul(old_U, old_U.T)
            cov_matrix = torch.matmul(P_old, torch.matmul(cov_matrix, P_old))

        # 2. SVD 分解
        U, S, V = torch.linalg.svd(cov_matrix)
        
        # 3. 按能量阈值截断
        total_var = torch.sum(S)
        curr_var = 0
        k = 0
        for i in range(len(S)):
            curr_var += S[i]
            if curr_var / (total_var + 1e-8) > threshold:
                k = i + 1
                break
        
        U_new = U[:, :k]

        # 4. 合并并【重新正交化 (QR 分解)】
        if old_U is not None:
            U_combined = torch.cat([old_U, U_new], dim=1)
            # 使用 QR 分解提取严格的正交基 (Q 矩阵)
            Q, R = torch.linalg.qr(U_combined, mode='reduced')
            U_combined = Q
        else:
            U_combined = U_new

        # 5. 计算绝对安全的梯度投影矩阵 P
        I = torch.eye(cov_matrix.shape[0], device=cov_matrix.device)
        P_new = I - torch.matmul(U_combined, U_combined.T)

        return U_combined, P_new


class IncrementalMoETrainer(Trainer):
    """Trainer for incremental learning with MoE"""

    def train(self):
        """Train then merge temporary incremental router into base router for clean checkpoints."""
        results = super().train()

        # Merge only for incremental tasks where new_router may exist
        if self.cfg.CONT.TASK > 0:
            # Unwrap DDP so we can access model attributes directly
            raw_model = self.model.module if hasattr(self.model, 'module') else self.model
            merged = self._merge_incremental_router(raw_model)
            if merged:
                print(f"Merged temporary router into base router for Task {self.cfg.CONT.TASK}")
                checkpointer = DetectionCheckpointer(self.model, save_dir=self.cfg.OUTPUT_DIR)
                checkpointer.save(f"model_task{self.cfg.CONT.TASK}_router_merged")
                print("Saved merged checkpoint for next incremental task loading")

        # ======【新增：训练完成后提取特征保存 SVD】======
        # 注意：每次训练完 (包含 Task 0) 都要收集！
        raw_model = self.model.module if hasattr(self.model, 'module') else self.model
        self.collect_and_save_svd(self.cfg, raw_model, self.cfg.CONT.TASK)
        # ================================================

        return results

    @staticmethod
    def _merge_incremental_router(model):
        """Merge temporary new_router into base router if present."""
        if not (hasattr(model, 'sem_seg_head') and hasattr(model.sem_seg_head, 'predictor')):
            return False

        predictor = model.sem_seg_head.predictor
        if not hasattr(predictor, 'transformer_ffn_layers'):
            return False

        last_ffn = predictor.transformer_ffn_layers[-1]
        if hasattr(last_ffn, 'fix_router') and hasattr(last_ffn, 'new_router') and last_ffn.new_router is not None:
            last_ffn.fix_router()
            return True

        return False

    @staticmethod
    def _log_moe_trainability_status(model, current_task, output_dir):
        """Print explicit MoE trainability checks for task0/incremental tasks."""
        if not (hasattr(model, 'sem_seg_head') and hasattr(model.sem_seg_head, 'predictor')):
            return

        predictor = model.sem_seg_head.predictor
        if not hasattr(predictor, 'transformer_ffn_layers'):
            return

        last_ffn = predictor.transformer_ffn_layers[-1]
        if not hasattr(last_ffn, 'experts'):
            return

        expert_trainable = []
        for i, expert in enumerate(last_ffn.experts):
            is_trainable = any(p.requires_grad for p in expert.parameters())
            expert_trainable.append((i, is_trainable))

        base_router_trainable = any(p.requires_grad for p in last_ffn.router.parameters()) if hasattr(last_ffn, 'router') else False
        new_router_trainable = False
        if hasattr(last_ffn, 'new_router') and last_ffn.new_router is not None:
            new_router_trainable = any(p.requires_grad for p in last_ffn.new_router.parameters())

        os.makedirs(output_dir, exist_ok=True)

        param1_file = os.path.join(output_dir, 'moecanshu.txt')

        with open(param1_file, 'w') as f:
            f.write(f"[MoE Check][Task {current_task}] experts trainable: {expert_trainable}\n")
            f.write(f"[MoE Check][Task {current_task}] base router trainable: {base_router_trainable}\n")
            f.write(f"[MoE Check][Task {current_task}] new router trainable: {new_router_trainable}\n")

            if current_task == 0:
                f.write("[MoE Check][Task 0] Expectation: base router trainable=True, new_router=False, initial experts trainable.\n")
            else:
                f.write("[MoE Check][Incremental] Expectation: old experts/base router frozen, only new expert + new_router trainable.\n")

    @classmethod
    def collect_and_save_svd(cls, cfg, model, current_task):
        svd_log_path = os.path.join(cfg.OUTPUT_DIR, "svd.txt")

        def log_svd(msg):
            if comm.is_main_process():
                with open(svd_log_path, "a") as f:
                    f.write(msg + "\n")

        log_svd(f"[SVD] 正在为 Task {current_task} 收集特征并计算全局正交基...")

        model.eval()
        svd_manager = SVDManager(feature_dim=cfg.MODEL.MASK_FORMER.HIDDEN_DIM)

        # 1. 挂载 Forward Hook (所有卡)
        h1 = model.sem_seg_head.predictor.class_embed.register_forward_hook(svd_manager.hook_frame)
        h2 = model.vita_module.class_embed.register_forward_hook(svd_manager.hook_clip)

        # 2. 构造 Dataloader 跑少量数据 (约 100 个 iter)
        data_loader = cls.build_train_loader(cfg)
        max_iters = max(1, 100 // comm.get_world_size())
        
        with torch.no_grad():
            for idx, batch in enumerate(data_loader):
                if idx >= max_iters:
                    break
                model(batch)
        
        h1.remove()
        h2.remove()

        # 3. 同步多卡协方差
        svd_manager.sync_covariance()

        # 4. 加载上一个任务的老正交基
        old_U_frame, old_U_clip = None, None
        if current_task > 0:
            prev_dir = f"{cfg.OUTPUT_DIR}/../step{current_task-1}"
            old_svd_path = os.path.join(prev_dir, f"svd_task{current_task-1}.pth")
            if os.path.exists(old_svd_path):
                old_svd = torch.load(old_svd_path, map_location="cpu")
                old_U_frame = old_svd["U_frame"].cuda()
                old_U_clip = old_svd["U_clip"].cuda()

        # 5. 更新基向量并计算投影矩阵 P
        svd_threshold = cfg.CONT.SVD_THRESHOLD
        log_svd(f"[SVD] 使用能量阈值 threshold={svd_threshold}")
        U_frame, P_frame = SVDManager.update_basis(old_U_frame, svd_manager.cov_frame, threshold=svd_threshold)
        U_clip, P_clip = SVDManager.update_basis(old_U_clip, svd_manager.cov_clip, threshold=svd_threshold)

        # 6. 主进程保存文件
        if comm.is_main_process():
            log_svd(f"[SVD] Frame 全局基向量累积维度: {U_frame.shape[1]}/256")
            log_svd(f"[SVD] Clip  全局基向量累积维度: {U_clip.shape[1]}/256")

            save_path = os.path.join(cfg.OUTPUT_DIR, f"svd_task{current_task}.pth")
            torch.save({
                "U_frame": U_frame.cpu(), "P_frame": P_frame.cpu(),
                "U_clip": U_clip.cpu(), "P_clip": P_clip.cpu()
            }, save_path)
            log_svd(f"[SVD] 全局 SVD 投影矩阵已保存至: {save_path}")
        
        # 强制同步，防止其他进程提前进入下一步
        comm.synchronize()
        

    @classmethod
    def build_model(cls, cfg):
        model = super().build_model(cfg)
        # 【新增这段代码】：专门为了 Task 0 加载 COCO 预训练权重
        if cfg.CONT.TASK == 0 and cfg.CONT.WEIGHTS:
            print(f"Loading Task 0 Base model from {cfg.CONT.WEIGHTS}")
            checkpointer = DetectionCheckpointer(model)
            checkpointer.load(cfg.CONT.WEIGHTS)
            # 加载完后，依然保持全部可训练，记录状态即可
            cls._log_moe_trainability_status(model, cfg.CONT.TASK, cfg.OUTPUT_DIR)

        # For Task 1+: Load previous task model and add exactly one new expert
        if cfg.CONT.TASK > 0 and cfg.CONT.WEIGHTS:
            print(f"Loading Task {cfg.CONT.TASK - 1} model from {cfg.CONT.WEIGHTS}")
            checkpointer = DetectionCheckpointer(model)
            checkpointer.load(cfg.CONT.WEIGHTS)

            # Add new expert to Frame Decoder (only once per incremental task)
            if hasattr(model, 'sem_seg_head') and hasattr(model.sem_seg_head, 'predictor'):
                predictor = model.sem_seg_head.predictor
                if hasattr(predictor, 'add_new_expert_for_task'):
                    before_cnt = -1
                    after_cnt = -1
                    if hasattr(predictor, 'transformer_ffn_layers'):
                        last_ffn = predictor.transformer_ffn_layers[-1]
                        if hasattr(last_ffn, 'experts'):
                            before_cnt = len(last_ffn.experts)

                    print(f"Adding new expert for Task {cfg.CONT.TASK}")
                    predictor.add_new_expert_for_task(cfg.CONT.TASK)

                    if hasattr(predictor, 'transformer_ffn_layers'):
                        last_ffn = predictor.transformer_ffn_layers[-1]
                        if hasattr(last_ffn, 'experts'):
                            after_cnt = len(last_ffn.experts)

                    expected_cnt = cfg.CONT.BASE_EXPERTS + cfg.CONT.TASK
                    print(f"Expert count: before={before_cnt}, after={after_cnt}, expected={expected_cnt}")

                    # Move new expert to the same device as the model
                    device = next(model.parameters()).device
                    model = model.to(device)
                    print(f"Model moved to device: {device}")

            # Freeze all shared components AFTER adding new expert, so only the new one is trainable
            cls._freeze_shared_components(
                model,
                cfg.CONT.TASK,
                cfg.CONT.BASE_EXPERTS,
                cfg.OUTPUT_DIR,
                cfg.CONT.BASE_CLS,
                cfg.CONT.INC_CLS,
            )
            cls._log_moe_trainability_status(model, cfg.CONT.TASK, cfg.OUTPUT_DIR)
        else:
            # Task 0 should keep base router and initial experts trainable
            cls._log_moe_trainability_status(model, cfg.CONT.TASK, cfg.OUTPUT_DIR)

        return model

    @staticmethod
    def _freeze_shared_components(model, current_task, base_experts, output_dir, base_cls, inc_cls):
        """Freeze all shared components for incremental learning"""
        import os

        os.makedirs(output_dir, exist_ok=True)

        # ======【新增：加载上一个任务的 SVD 投影矩阵】======
        num_old_classes = (
            base_cls + (current_task - 1) * inc_cls
            if current_task > 0 else base_cls
        )
        P_frame, P_clip = None, None
        if current_task > 0:
            prev_dir = f"{output_dir}/../step{current_task-1}"
            svd_path = os.path.join(prev_dir, f"svd_task{current_task-1}.pth")
            if os.path.exists(svd_path):
                device = next(model.parameters()).device
                svd_data = torch.load(svd_path, map_location=device)
                P_frame = svd_data["P_frame"]
                P_clip = svd_data["P_clip"]
            
            # 权重投影 Hook (保护 W)
            def get_weight_hook(P_matrix):
                def hook(grad):
                    return torch.matmul(grad, P_matrix)
                return hook

            # 偏置掩码 Hook (保护 b)
            def get_bias_hook(num_old_cls):
                def hook(grad):
                    grad_clone = grad.clone()
                    # 强行将老类别的梯度清零，不让老类别 Bias 发生任何偏移
                    grad_clone[:num_old_cls] = 0.0 
                    return grad_clone
                return hook

        # Open file for writing parameter info
        param_file = os.path.join(output_dir, 'canshu.txt')

        with open(param_file, 'w') as f:
            f.write(f"{'='*60}\n")
            f.write(f"Freezing shared components for Task {current_task}\n")
            f.write(f"{'='*60}\n\n")

            # Step 1: Freeze ALL parameters
            for name, param in model.named_parameters():
                param.requires_grad = False
            f.write("✓ All parameters frozen\n\n")

            # Step 2: Unfreeze ONLY trainable components
            f.write("Unfreezing trainable components:\n")

            # Unfreeze Query Embeddings
            if hasattr(model, 'sem_seg_head') and hasattr(model.sem_seg_head, 'predictor'):
                predictor = model.sem_seg_head.predictor
                # if hasattr(predictor, 'query_feat'):
                #     predictor.query_feat.weight.requires_grad = True
                #     f.write("  ✓ query_feat unfrozen\n")
                # if hasattr(predictor, 'query_embed'):
                #     predictor.query_embed.weight.requires_grad = True
                #     f.write("  ✓ query_embed unfrozen\n")

                # Unfreeze Prediction Heads
                        
                    # ======= 针对 Frame 头 =======
                if hasattr(predictor, 'class_embed'):
                    for name, param in predictor.class_embed.named_parameters():
                        param.requires_grad = True
                    
                        if current_task > 0 and P_frame is not None:
                            if 'weight' in name:
                                param.register_hook(get_weight_hook(P_frame))
                                f.write("  ✓ predictor.class_embed.weight SVD hooked!\n")
                            elif 'bias' in name:
                                param.register_hook(get_bias_hook(num_old_classes))
                                f.write("  ✓ predictor.class_embed.bias Mask hooked!\n")
                # if hasattr(predictor, 'mask_embed'):
                #     for param in predictor.mask_embed.parameters():
                #         param.requires_grad = True
                #     f.write("  ✓ mask_embed unfrozen\n")

                # Unfreeze ONLY the new Expert in MoE layer
                if hasattr(predictor, 'transformer_ffn_layers'):
                    last_ffn = predictor.transformer_ffn_layers[-1]
                    # Check if it's MoE layer
                    if hasattr(last_ffn, 'experts') and hasattr(last_ffn, 'num_experts'):
                        # With BASE_EXPERTS base experts, incremental task t adds expert at index (BASE_EXPERTS + t - 1)
                        # Only unfreeze this newly added expert.
                        new_expert_idx = base_experts + current_task - 1
                        if new_expert_idx < len(last_ffn.experts):
                            for param in last_ffn.experts[new_expert_idx].parameters():
                                param.requires_grad = True
                                
                            f.write(f"  ✓ Expert {new_expert_idx} (new expert) unfrozen\n")

                            # Verify old experts are frozen
                            for i in range(new_expert_idx):
                                for param in last_ffn.experts[i].parameters():
                                    if param.requires_grad:
                                        f.write(f"  ⚠ WARNING: Expert {i} is NOT frozen!\n")

                            # Verify base router is frozen
                            router_frozen = True
                            for param in last_ffn.router.parameters():
                                if param.requires_grad:
                                    router_frozen = False
                                    f.write(f"  ⚠ WARNING: Base router is NOT frozen!\n")
                            if router_frozen:
                                f.write(f"  ✓ Base router is frozen (as expected)\n")

                            # Unfreeze temporary new_router in incremental task
                            if hasattr(last_ffn, 'new_router') and last_ffn.new_router is not None:
                                for param in last_ffn.new_router.parameters():
                                    param.requires_grad = True
                                f.write(f"  ✓ New router is unfrozen and trainable\n")

                            # Verify temporary new_router is trainable in incremental task
                            if hasattr(last_ffn, 'new_router') and last_ffn.new_router is not None:
                                new_router_trainable = True
                                for param in last_ffn.new_router.parameters():
                                    if not param.requires_grad:
                                        new_router_trainable = False

                                        f.write(f"  ⚠ WARNING: New router is frozen unexpectedly!\n")
                                if new_router_trainable:
                                    f.write(f"  ✓ New router is trainable (as expected)\n")

            # ======= 针对 Clip (VITA) 头 =======
            if hasattr(model, 'vita_module'):
                vita = model.vita_module
                if hasattr(vita, 'class_embed'):
                    for name, param in vita.class_embed.named_parameters():
                        param.requires_grad = True
                        
                        if current_task > 0 and P_clip is not None:
                            if 'weight' in name:
                                param.register_hook(get_weight_hook(P_clip))
                                f.write("  ✓ vita_module.class_embed.weight SVD hooked!\n")
                            elif 'bias' in name:
                                param.register_hook(get_bias_hook(num_old_classes))
                                f.write("  ✓ vita_module.class_embed.bias Mask hooked!\n")
                # if hasattr(vita, 'mask_embed'):
                #     for param in vita.mask_embed.parameters():
                #         param.requires_grad = True
                    # f.write("  ✓ vita_module.mask_embed unfrozen\n")
                # if hasattr(vita, 'query_feat'):
                #     vita.query_feat.weight.requires_grad = True
                #     f.write("  ✓ vita_module query_feat unfrozen\n")
                # if hasattr(vita, 'query_embed'):
                #     vita.query_embed.weight.requires_grad = True
                #     f.write("  ✓ vita_module query_embed unfrozen\n")

            # Print trainable parameters summary
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            f.write(f"\n{'='*60}\n")
            f.write(f"Total parameters: {total_params:,}\n")
            f.write(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)\n")
            f.write(f"Frozen parameters: {total_params-trainable_params:,} ({100*(total_params-trainable_params)/total_params:.2f}%)\n")
            f.write(f"{'='*60}\n\n")

            # List all trainable parameters
            f.write("Trainable parameter names:\n")
            for name, param in model.named_parameters():
                if param.requires_grad:
                    f.write(f"  - {name}: {param.numel():,} params\n")

            # List all untrainable parameters
            f.write("unTrainable parameter names:\n")
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    f.write(f"  - {name}: {param.numel():,} params\n")

        # Also print to console
        # print(f"Parameter info saved to: {param_file}")
        # print(f"Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")


def setup(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_vita_config(cfg)
    add_continual_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = IncrementalMoETrainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = IncrementalMoETrainer.test(cfg, model)
        return res

    trainer = IncrementalMoETrainer(cfg)
    #trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

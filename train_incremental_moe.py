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


class IncrementalMoETrainer(Trainer):
    """Trainer for incremental learning with MoE"""

    @classmethod
    def build_model(cls, cfg):
        model = super().build_model(cfg)

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

                    print(f"Expert count: before={before_cnt}, after={after_cnt}, expected={cfg.CONT.TASK + 1}")

                    # Move new expert to the same device as the model
                    device = next(model.parameters()).device
                    model = model.to(device)
                    print(f"Model moved to device: {device}")

            # Freeze all shared components AFTER adding new expert, so only the new one is trainable
            cls._freeze_shared_components(model, cfg.CONT.TASK)

        return model

    @staticmethod
    def _freeze_shared_components(model, current_task):
        """Freeze all shared components for incremental learning"""
        import os

        # Get output directory from model's config
        output_dir = model.cfg.OUTPUT_DIR if hasattr(model, 'cfg') else 'output'
        os.makedirs(output_dir, exist_ok=True)

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
                # if hasattr(predictor, 'class_embed'):
                #     for param in predictor.class_embed.parameters():
                #         param.requires_grad = True
                #     f.write("  ✓ class_embed unfrozen\n")
                # if hasattr(predictor, 'mask_embed'):
                #     for param in predictor.mask_embed.parameters():
                #         param.requires_grad = True
                #     f.write("  ✓ mask_embed unfrozen\n")

                # Unfreeze ONLY the new Expert in MoE layer
                if hasattr(predictor, 'transformer_ffn_layers'):
                    last_ffn = predictor.transformer_ffn_layers[-1]
                    # Check if it's MoE layer
                    if hasattr(last_ffn, 'experts') and hasattr(last_ffn, 'num_experts'):
                        # Only unfreeze the last expert (new expert for current task)
                        new_expert_idx = current_task
                        if new_expert_idx < len(last_ffn.experts):
                            for param in last_ffn.experts[new_expert_idx].parameters():
                                param.requires_grad = True
                            f.write(f"  ✓ Expert {new_expert_idx} (new expert) unfrozen\n")

                            # Verify old experts are frozen
                            for i in range(new_expert_idx):
                                for param in last_ffn.experts[i].parameters():
                                    if param.requires_grad:
                                        f.write(f"  ⚠ WARNING: Expert {i} is NOT frozen!\n")

                            # Verify router is frozen
                            router_frozen = True
                            for param in last_ffn.router.parameters():
                                if param.requires_grad:
                                    router_frozen = False
                                    f.write(f"  ⚠ WARNING: Router is NOT frozen!\n")
                            if router_frozen:
                                f.write(f"  ✓ Router is frozen (as expected)\n")

            # Unfreeze VITA module prediction heads
            # if hasattr(model, 'vita_module'):
            #     vita = model.vita_module
            #     if hasattr(vita, 'class_embed'):
            #         for param in vita.class_embed.parameters():
            #             param.requires_grad = True
            #         f.write("  ✓ vita_module.class_embed unfrozen\n")
                # if hasattr(vita, 'mask_embed'):
                #     for param in vita.mask_embed.parameters():
                #         param.requires_grad = True
                #     f.write("  ✓ vita_module.mask_embed unfrozen\n")

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

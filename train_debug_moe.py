"""
Debug Training Script - Use GT masks to isolate MoE issues

This script freezes mask prediction and uses GT masks to debug MoE behavior
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from train_incremental_moe import IncrementalMoETrainer, setup, default_argument_parser, launch
from detectron2.checkpoint import DetectionCheckpointer
import torch


class DebugMoETrainer(IncrementalMoETrainer):
    """Debug trainer that uses GT masks"""

    @classmethod
    def build_model(cls, cfg):
        model = super().build_model(cfg)

        # Additional freezing for debug: freeze mask prediction
        if cfg.CONT.TASK > 0:
            print("\n" + "="*60)
            print("DEBUG MODE: Freezing mask prediction components")
            print("="*60)

            # Freeze mask_embed in Frame Decoder
            if hasattr(model, 'sem_seg_head') and hasattr(model.sem_seg_head, 'predictor'):
                predictor = model.sem_seg_head.predictor
                if hasattr(predictor, 'mask_embed'):
                    for param in predictor.mask_embed.parameters():
                        param.requires_grad = False
                    print("✓ Frame Decoder mask_embed frozen")

            # Freeze mask_embed in VITA module
            if hasattr(model, 'vita_module'):
                vita = model.vita_module
                if hasattr(vita, 'mask_embed'):
                    for param in vita.mask_embed.parameters():
                        param.requires_grad = False
                    print("✓ VITA module mask_embed frozen")

            # Print final trainable components
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"\nTrainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
            print("="*60 + "\n")

        return model


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = DebugMoETrainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = DebugMoETrainer.test(cfg, model)
        return res

    trainer = DebugMoETrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    print("\n" + "="*60)
    print("DEBUG MODE: Mask prediction frozen, only training:")
    print("  - Frame Decoder: class_embed + new Expert")
    print("  - VITA module: class_embed")
    print("="*60 + "\n")
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )


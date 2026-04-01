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

        # For Task 1+: Load previous task model and add new expert
        if cfg.CONT.TASK > 0 and cfg.CONT.WEIGHTS:
            print(f"Loading Task {cfg.CONT.TASK - 1} model from {cfg.CONT.WEIGHTS}")
            checkpointer = DetectionCheckpointer(model)
            checkpointer.load(cfg.CONT.WEIGHTS)

            # Freeze all shared components
            cls._freeze_shared_components(model, cfg.CONT.TASK)

            # Add new expert to Frame Decoder
            if hasattr(model, 'sem_seg_head') and hasattr(model.sem_seg_head, 'predictor'):
                predictor = model.sem_seg_head.predictor
                if hasattr(predictor, 'add_new_expert_for_task'):
                    print(f"Adding new expert for Task {cfg.CONT.TASK}")
                    predictor.add_new_expert_for_task(cfg.CONT.TASK)

                    # Move new expert to the same device as the model
                    device = next(model.parameters()).device
                    model = model.to(device)
                    print(f"Model moved to device: {device}")

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

            # 1. Freeze Backbone
            if hasattr(model, 'backbone'):
                for name, param in model.backbone.named_parameters():
                    param.requires_grad = False
                f.write("✓ Backbone frozen\n")

            # 2. Freeze Pixel Decoder
            if hasattr(model, 'sem_seg_head') and hasattr(model.sem_seg_head, 'pixel_decoder'):
                for name, param in model.sem_seg_head.pixel_decoder.named_parameters():
                    param.requires_grad = False
                f.write("✓ Pixel Decoder frozen\n")

            # 3. Freeze Frame Decoder shared layers (all except last MoE layer)
            if hasattr(model, 'sem_seg_head') and hasattr(model.sem_seg_head, 'predictor'):
                predictor = model.sem_seg_head.predictor

                # Freeze self-attention layers
                if hasattr(predictor, 'transformer_self_attention_layers'):
                    for layer in predictor.transformer_self_attention_layers:
                        for param in layer.parameters():
                            param.requires_grad = False

                # Freeze cross-attention layers
                if hasattr(predictor, 'transformer_cross_attention_layers'):
                    for layer in predictor.transformer_cross_attention_layers:
                        for param in layer.parameters():
                            param.requires_grad = False

                # Freeze FFN layers except the last one (MoE layer)
                if hasattr(predictor, 'transformer_ffn_layers'):
                    for i, layer in enumerate(predictor.transformer_ffn_layers):
                        if i < len(predictor.transformer_ffn_layers) - 1:
                            # Freeze non-MoE FFN layers
                            for param in layer.parameters():
                                param.requires_grad = False

                # Freeze other components (except query embeddings and prediction heads)
                if hasattr(predictor, 'level_embed'):
                    predictor.level_embed.weight.requires_grad = False
                if hasattr(predictor, 'input_proj'):
                    for proj in predictor.input_proj:
                        for param in proj.parameters():
                            param.requires_grad = False
                if hasattr(predictor, 'decoder_norm'):
                    for param in predictor.decoder_norm.parameters():
                        param.requires_grad = False

                f.write("✓ Frame Decoder shared layers frozen\n")
                f.write("✓ Query Embeddings kept trainable\n")
                f.write("✓ Prediction Heads kept trainable\n")

            # Print trainable parameters summary
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            f.write(f"\n{'='*60}\n")
            f.write(f"Total parameters: {total_params:,}\n")
            f.write(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)\n")
            f.write(f"Frozen parameters: {total_params-trainable_params:,} ({100*(total_params-trainable_params)/total_params:.2f}%)\n")
            f.write(f"{'='*60}\n")

        # Also print to console
        print(f"Parameter info saved to: {param_file}")


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
    trainer.resume_or_load(resume=args.resume)
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

"""
Modified VITA model that uses GT masks for debugging
"""
import torch
from vita.vita_model import Vita
from detectron2.modeling import META_ARCH_REGISTRY


@META_ARCH_REGISTRY.register()
class VitaDebugGTMask(Vita):
    """VITA model that uses GT masks instead of predicted masks for debugging"""

    def train_model(self, batched_inputs):
        images = []
        for video in batched_inputs:
            for frame in video["image"]:
                images.append(frame.to(self.device))
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        from detectron2.structures import ImageList
        images = ImageList.from_tensors(images, self.size_divisibility)
        features = self.backbone(images.tensor)

        BT = len(images)
        T = self.num_frames if self.training else BT
        B = BT // T

        outputs, frame_queries, mask_features = self.sem_seg_head(features)

        mask_features = self.vita_module.vita_mask_features(mask_features)
        mask_features = mask_features.view(B, self.num_frames, *mask_features.shape[-3:])

        # mask classification target
        frame_targets, clip_targets = self.prepare_targets(batched_inputs, images)

        # bipartite matching-based loss
        losses, fg_indices = self.criterion(outputs, frame_targets)

        vita_outputs = self.vita_module(frame_queries)

        # DEBUG: Use GT masks instead of predicted masks
        # Expected shape: [L, B, cQ, T, H, W]

        # Get number of decoder layers
        L = len(vita_outputs["aux_outputs"]) + 1

        # Extract GT masks
        gt_masks_batch = []
        for clip_target in clip_targets:
            gt_masks_batch.append(clip_target['masks'].float())

        T = self.num_frames
        H, W = gt_masks_batch[0].shape[-2:]

        # Pad to same number of queries (cQ)
        cQ = vita_outputs["pred_mask_embed"].shape[2]
        padded_masks = []
        for gt_mask in gt_masks_batch:
            N = gt_mask.shape[0]
            if N < cQ:
                pad_mask = torch.zeros(cQ - N, T, H, W, device=gt_mask.device, dtype=gt_mask.dtype)
                gt_mask = torch.cat([gt_mask, pad_mask], dim=0)
            else:
                gt_mask = gt_mask[:cQ]
            padded_masks.append(gt_mask)

        # Stack to [B, cQ, T, H, W]
        gt_masks_tensor = torch.stack(padded_masks, dim=0)

        # Expand to [L, B, cQ, T, H, W]
        gt_masks_tensor = gt_masks_tensor.unsqueeze(0).expand(L, -1, -1, -1, -1, -1)

        vita_outputs["pred_masks"] = gt_masks_tensor

        # Fix aux_outputs shape - each should be [L, B, cQ, T, H, W] not [B, cQ, T, H, W]
        for i, out in enumerate(vita_outputs["aux_outputs"]):
            out["pred_masks"] = gt_masks_tensor  # Use same tensor for all aux outputs

        for k in list(losses.keys()):
            if k in self.criterion.weight_dict:
                losses[k] *= self.criterion.weight_dict[k]
            else:
                losses.pop(k)

        vita_loss_dict = self.vita_criterion(vita_outputs, clip_targets, frame_targets, fg_indices)
        vita_weight_dict = self.vita_criterion.weight_dict

        for k in vita_loss_dict.keys():
            if k in vita_weight_dict:
                vita_loss_dict[k] *= vita_weight_dict[k]
        losses.update(vita_loss_dict)
        return losses


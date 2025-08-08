# scripts/loss.py

import torch
import torch.nn.functional as F
from torch import nn
from scipy.optimize import linear_sum_assignment

import sys
sys.path.append('e:/version_2/vit_detector')
from scripts.utils import box_cxcywh_to_xyxy

class HungarianMatcher(nn.Module):
    """
    This class computes an assignment between the targets and the predictions of the network.
    For efficiency reasons, the cost is computed using the out-of-loop top-k predictions.
    """
    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)
        out_bbox = outputs["pred_boxes"].flatten(0, 1)

        # Also concat the target labels and boxes
        valid_targets = [v for v in targets if len(v["labels"]) > 0 and len(v["boxes"]) > 0]
        
        # Handle case where no valid targets exist
        if len(valid_targets) == 0:
            return [(torch.as_tensor([], dtype=torch.int64), torch.as_tensor([], dtype=torch.int64)) for _ in range(bs)]
        
        tgt_ids = torch.cat([v["labels"] for v in valid_targets])
        tgt_bbox = torch.cat([v["boxes"] for v in valid_targets])

        # Compute the classification cost.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost between boxes
        # This requires converting to xyxy format
        out_bbox_xyxy = box_cxcywh_to_xyxy(out_bbox)
        tgt_bbox_xyxy = box_cxcywh_to_xyxy(tgt_bbox)
        
        # Calculate GIoU cost efficiently
        cost_giou = []
        for out_box in out_bbox_xyxy:
            giou_row = []
            for tgt_box in tgt_bbox_xyxy:
                # Calculate intersection area
                inter_min = torch.max(out_box[:2], tgt_box[:2])
                inter_max = torch.min(out_box[2:], tgt_box[2:])
                inter_wh = (inter_max - inter_min).clamp(min=0)
                inter_area = inter_wh[0] * inter_wh[1]

                # Calculate union area
                out_area = (out_box[2] - out_box[0]) * (out_box[3] - out_box[1])
                tgt_area = (tgt_box[2] - tgt_box[0]) * (tgt_box[3] - tgt_box[1])
                union_area = out_area + tgt_area - inter_area

                iou = inter_area / (union_area + 1e-6)
                
                # Calculate enclosing box
                enclose_min = torch.min(out_box[:2], tgt_box[:2])
                enclose_max = torch.max(out_box[2:], tgt_box[2:])
                enclose_wh = (enclose_max - enclose_min).clamp(min=0)
                enclose_area = enclose_wh[0] * enclose_wh[1]
                
                giou = iou - (enclose_area - union_area) / (enclose_area + 1e-6)
                giou_row.append(-giou)  # Negative because we want cost
            cost_giou.append(torch.stack(giou_row))
        cost_giou = torch.stack(cost_giou)


        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        
        # Handle empty targets by creating proper assignments
        if all(s == 0 for s in sizes):
            return [(torch.as_tensor([], dtype=torch.int64), torch.as_tensor([], dtype=torch.int64)) for _ in range(bs)]
        
        # For non-empty targets, split and assign
        cost_splits = C.split(sizes, -1)
        indices = []
        for i, (c, size) in enumerate(zip(cost_splits, sizes)):
            if size == 0:
                indices.append((torch.as_tensor([], dtype=torch.int64), torch.as_tensor([], dtype=torch.int64)))
            else:
                assignment = linear_sum_assignment(c[i])
                indices.append((torch.as_tensor(assignment[0], dtype=torch.int64), torch.as_tensor(assignment[1], dtype=torch.int64)))
        
        return indices


class SetCriterion(nn.Module):
    """
    This class computes the loss for DETR.
    The process happens in two steps:
    1) we compute hungarian assignment between ground truth boxes and the outputs of the model
    2) we supervise each pair of matched ground-truth / prediction
    """
    def __init__(self, num_classes, matcher, eos_coef, losses, weight_dict=None):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.eos_coef = eos_coef
        self.losses = losses
        self.weight_dict = weight_dict or {}
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        
        # Handle empty targets case
        target_classes_list = [t["labels"][J] for t, (_, J) in zip(targets, indices) if len(J) > 0]
        if not target_classes_list:
            target_classes_o = torch.tensor([], dtype=torch.int64, device=src_logits.device)
        else:
            target_classes_o = torch.cat(target_classes_list)
            
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        
        if len(target_classes_o) > 0:
            target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        
        # Handle empty targets case
        target_boxes_list = [t['boxes'][i] for t, (_, i) in zip(targets, indices) if len(i) > 0]
        if not target_boxes_list:
            # No targets, return zero losses
            losses = {
                'loss_bbox': torch.tensor(0.0, device=outputs['pred_boxes'].device, requires_grad=True),
                'loss_giou': torch.tensor(0.0, device=outputs['pred_boxes'].device, requires_grad=True)
            }
            return losses
            
        target_boxes = torch.cat(target_boxes_list, dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        
        # GIoU Loss
        # This is a simplified version. A library like torchvision.ops.generalized_box_iou would be better.
        src_boxes_xyxy = box_cxcywh_to_xyxy(src_boxes)
        target_boxes_xyxy = box_cxcywh_to_xyxy(target_boxes)
        
        # Calculate intersection area
        inter_mins = torch.max(src_boxes_xyxy[:, :2], target_boxes_xyxy[:, :2])
        inter_maxs = torch.min(src_boxes_xyxy[:, 2:], target_boxes_xyxy[:, 2:])
        inter_wh = (inter_maxs - inter_mins).clamp(min=0)
        inter_area = inter_wh[:, 0] * inter_wh[:, 1]

        # Calculate union area
        src_area = (src_boxes_xyxy[:, 2] - src_boxes_xyxy[:, 0]) * (src_boxes_xyxy[:, 3] - src_boxes_xyxy[:, 1])
        tgt_area = (target_boxes_xyxy[:, 2] - target_boxes_xyxy[:, 0]) * (target_boxes_xyxy[:, 3] - target_boxes_xyxy[:, 1])
        union_area = src_area + tgt_area - inter_area

        iou = inter_area / (union_area + 1e-6)
        
        # Calculate enclosing box
        enclose_mins = torch.min(src_boxes_xyxy[:, :2], target_boxes_xyxy[:, :2])
        enclose_maxs = torch.max(src_boxes_xyxy[:, 2:], target_boxes_xyxy[:, 2:])
        enclose_wh = (enclose_maxs - enclose_mins).clamp(min=0)
        enclose_area = enclose_wh[:, 0] * enclose_wh[:, 1]
        
        giou = iou - (enclose_area - union_area) / (enclose_area + 1e-6)
        
        loss_giou = 1 - torch.diag(giou) # This is not quite right, needs proper broadcasting
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        return losses

    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes"""
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'loss_cardinality': card_err}
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'cardinality': self.loss_cardinality,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes)

    def forward(self, outputs, targets):
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # Apply loss weights if provided
        if self.weight_dict:
            weighted_losses = {}
            for k, v in losses.items():
                if k in self.weight_dict:
                    weighted_losses[k] = v * self.weight_dict[k]
                else:
                    weighted_losses[k] = v
            return weighted_losses
        
        return losses

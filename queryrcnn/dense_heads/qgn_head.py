import math
from scipy.optimize import linear_sum_assignment

import torch
import torch.nn.functional as F
from torch import nn
from typing import List
from fvcore.nn import sigmoid_focal_loss_jit

from detectron2.layers import cat
from detectron2.structures import Boxes, Instances, pairwise_iou

from ..util.misc import (get_world_size, is_dist_avail_and_initialized)
from ..util import box_ops


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


class StaRPNHead(nn.Module):
    """
    Standard RPN classification and regression heads described in :paper:`Faster R-CNN`.
    Uses a 3x3 conv to produce a shared hidden state from which one 1x1 conv predicts
    objectness logits for each anchor and a second 1x1 conv predicts bounding-box deltas
    specifying how to deform each anchor into an object proposal.
    """

    def __init__(self, cfg, input_shape, box_dim: int = 4):
        super().__init__()
        # 3x3 conv for the hidden representation
        in_channels = input_shape['p3'].channels
        self.num_classes = cfg.MODEL.QueryRCNN.RPN.NUM_CLASSES
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        # 1x1 conv for predicting objectness logits
        self.objectness_logits = nn.Conv2d(in_channels, self.num_classes, kernel_size=1, stride=1)
        self.proposal_feats = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
        self.anchor_deltas = nn.Conv2d(in_channels, box_dim, kernel_size=1, stride=1)

        for l in [self.conv, self.objectness_logits, self.anchor_deltas]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)

        self.fpn_strides = cfg.MODEL.QueryRCNN.RPN.FPN_STRIDES
        self.norm_reg_targets = True
        self.scales = nn.ModuleList(
            [Scale(init_value=1.0) for _ in range(len(self.fpn_strides))])

        bias_value = -math.log((1 - 0.01) / 0.01)
        torch.nn.init.constant_(self.objectness_logits.bias, bias_value)

    def forward(self, features: List[torch.Tensor]):
        """
        Args:
            features (list[Tensor]): list of feature maps

        Returns:
            list[Tensor]: A list of L elements.
                Element i is a tensor of shape (N, A, Hi, Wi) representing
                the predicted objectness logits for all anchors. A is the number of cell anchors.
            list[Tensor]: A list of L elements. Element i is a tensor of shape
                (N, A*box_dim, Hi, Wi) representing the predicted "deltas" used to transform anchors
                to proposals.
        """
        pred_objectness_logits = []
        pred_anchor_deltas = []
        feats = []
        # filter_subnet = []
        for level, x in enumerate(features):
            t = F.relu(self.conv(x))
            bbox_pred = self.scales[level](self.anchor_deltas(t))
            if self.norm_reg_targets:
                pred_anchor_deltas.append(F.relu(bbox_pred) * self.fpn_strides[level])
            else:
                pred_anchor_deltas.append(torch.exp(bbox_pred))
            # filter_subnet.append(t)
            pred_objectness_logits.append(self.objectness_logits(t))
            feats.append(self.proposal_feats(t))

        return pred_objectness_logits, feats, pred_anchor_deltas


class QGN(nn.Module):
    """
    Standard RPN classification and regression heads described in :paper:`Faster R-CNN`.
    Uses a 3x3 conv to produce a shared hidden state from which one 1x1 conv predicts
    objectness logits for each anchor and a second 1x1 conv predicts bounding-box deltas
    specifying how to deform each anchor into an object proposal.
    """

    def __init__(self, cfg, input_shape, in_features, box_dim: int = 4):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels (int): number of input feature channels. When using multiple
                input features, they must have the same number of channels.
            num_anchors (int): number of anchors to predict for *each spatial position*
                on the feature map. The total number of anchors for each
                feature map will be `num_anchors * H * W`.
            box_dim (int): dimension of a box, which is also the number of box regression
                predictions to make for each anchor. An axis aligned box has
                box_dim=4, while a rotated box has box_dim=5.
        """
        super().__init__()
        self.in_features = cfg.MODEL.RPN.IN_FEATURES
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.rpn_head = StaRPNHead(cfg, input_shape, box_dim=4)
        self.box_dim = box_dim
        self.fpn_strides = cfg.MODEL.QueryRCNN.RPN.FPN_STRIDES
        self.num_classes = cfg.MODEL.QueryRCNN.RPN.NUM_CLASSES
        self.topk_candidates = cfg.MODEL.QueryRCNN.RPN.TOPK_INDICES
        self.max_detections_per_image = cfg.MODEL.SparseRCNN.NUM_PROPOSALS
        self.nms_type = None

        self.objectness_alpha = cfg.MODEL.QueryRCNN.RPN.OJT_ALPHA
        self.cls_weight = cfg.MODEL.QueryRCNN.RPN.CLS_WEIGHT
        self.reg_weight = cfg.MODEL.QueryRCNN.RPN.REG_WEIGHT

    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = self.compute_locations_per_level(
                h, w, self.fpn_strides[level],
                feature.device
            )
            locations.append(locations_per_level)
        return locations

    def compute_locations_per_level(self, h, w, stride, device):
        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        return locations

    def apply_deltas(self, deltas, shifts):
        """
        Apply transformation `deltas` (dl, dt, dr, db) to `shifts`.
        Args:
            deltas (Tensor): transformation deltas of shape (N, k*4), where k >= 1.
                deltas[i] represents k potentially different class-specific
                box transformations for the single shift shifts[i].
            shifts (Tensor): shifts to transform, of shape (N, 2)
        """
        # import pdb; pdb.set_trace()
        # assert torch.isfinite(deltas).all().item()
        shifts = shifts.to(deltas.dtype)
        if deltas.numel() == 0:
            return torch.empty_like(deltas)

        deltas = deltas.view(deltas.size()[:-1] + (-1, 4))
        boxes = torch.cat((shifts.unsqueeze(-2) - deltas[..., :2],
                           shifts.unsqueeze(-2) + deltas[..., 2:]),
                          dim=-1).view(deltas.size()[:-2] + (-1, ))
        return boxes

    def get_deltas(self, shifts, boxes):
        """
        Get box regression transformation deltas (dl, dt, dr, db) that can be used
        to transform the `shifts` into the `boxes`. That is, the relation
        ``boxes == self.apply_deltas(deltas, shifts)`` is true.
        Args:
            shifts (Tensor): shifts, e.g., feature map coordinates
            boxes (Tensor): target of the transformation, e.g., ground-truth
                boxes.
        """
        assert isinstance(shifts, torch.Tensor), type(shifts)
        assert isinstance(boxes, torch.Tensor), type(boxes)

        deltas = torch.cat((shifts - boxes[..., :2], boxes[..., 2:] - shifts),
                           dim=-1)
        return deltas

    @torch.no_grad()
    def get_ground_truth(self, images, shifts, targets, box_cls, box_delta, filters=None):
        """
        Args:
            shifts (list[list[Tensor]]): a liget_ground_truth of N=#image elements. Each is a
                list of #feature level tensors. The tensors contains shifts of
                this image on the specific feature level.
            targets (list[Instances]): a list of N `Instances`s. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.  Specify `targets` during training only.

        Returns:
            gt_classes (Tensor):
                An integer tensor of shape (N, R) storing ground-truth
                labels for each shift.
                R is the total number of shifts, i.e. the sum of Hi x Wi for all levels.
                Shifts in the valid boxes are assigned their corresponding label in the
                [0, K-1] range. Shifts in the background are assigned the label "K".
                Shifts in the ignore areas are assigned a label "-1", i.e. ignore.
            gt_shifts_deltas (Tensor):
                Shape (N, R, 4).
                The last dimension represents ground-truth shift2box transform
                targets (dl, dt, dr, db) that map each shift to its matched ground-truth box.
                The values in the tensor are meaningful only when the corresponding
                shift is labeled as foreground.
        """
        gt_classes = []
        gt_shifts_deltas = []

        box_cls = torch.cat(box_cls, dim=1)
        box_delta = torch.cat(box_delta, dim=1)
        # filters = torch.cat(filters, dim=1)
        box_cls = box_cls.sigmoid_()
        # filters = filters.sigmoid_()
        num_fg = 0
        num_gt = 0
        img_idx = 0
        for targets_per_image, box_cls_per_image, box_delta_per_image in zip(
                targets, box_cls, box_delta):
            shifts_over_all_feature_maps = torch.cat(shifts, dim=0)

            gt_boxes = targets_per_image['boxes_xyxy'].to(box_delta.device)
            gt_boxes = Boxes(gt_boxes)
            if self.num_classes == 1:
                prob = box_cls_per_image.t()
            else:
                prob = box_cls_per_image[:, targets_per_image['labels']].t()
            boxes = self.apply_deltas(
                box_delta_per_image, shifts_over_all_feature_maps
            )
            iou = pairwise_iou(gt_boxes, Boxes(boxes))
            quality = prob ** self.objectness_alpha * iou ** (1 - self.objectness_alpha)

            deltas = self.get_deltas(
                shifts_over_all_feature_maps, gt_boxes.tensor.unsqueeze(1))

            if deltas.shape[0]:
                is_in_boxes = deltas.min(dim=-1).values > 0
            else:
                is_in_boxes = deltas.sum(dim=-1) > 0

            quality[~is_in_boxes] = -1
            # self.check_in_bbox(images, is_in_boxes, shifts_over_all_feature_maps)
            ins_nums = targets_per_image['labels'].shape[0]
            gt_idxs, shift_idxs = linear_sum_assignment(quality.cpu().numpy(), maximize=True)
            num_fg += len(shift_idxs)
            num_gt += ins_nums

            full_value = 0 if self.num_classes == 1 else self.num_classes
            gt_classes_i = shifts_over_all_feature_maps.new_full(
                (len(shifts_over_all_feature_maps),), full_value, dtype=torch.long
            )
            gt_shifts_reg_deltas_i = shifts_over_all_feature_maps.new_zeros(
                len(shifts_over_all_feature_maps), 4
            )
            if ins_nums > 0:
                # ground truth classes
                if self.num_classes == 1:
                    gt_classes_i[shift_idxs] = 1
                else:
                    gt_classes_i[shift_idxs] = targets_per_image['labels'][gt_idxs]
                # ground truth box regression
                gt_shifts_reg_deltas_i[shift_idxs] = self.get_deltas(
                    shifts_over_all_feature_maps[shift_idxs], gt_boxes[gt_idxs].tensor
                )
                
            gt_classes.append(gt_classes_i)
            gt_shifts_deltas.append(gt_shifts_reg_deltas_i)
            img_idx += 1

        return torch.stack(gt_classes), torch.stack(gt_shifts_deltas)

    def forward(
            self,
            images,
            features,
            targets,):
        """
        Args:
            features (list[Tensor]): list of feature maps

        Returns:
            list[Tensor]: A list of L elements.
                Element i is a tensor of shape (N, A, Hi, Wi) representing
                the predicted objectness logits for all anchors. A is the number of cell anchors.
            list[Tensor]: A list of L elements. Element i is a tensor of shape
                (N, A*box_dim, Hi, Wi) representing the predicted "deltas" used to transform anchors
                to proposals.
        """
        features = [features[f] for f in self.in_features]
        image_sizes = images.image_sizes  # h, w
        locations = self.compute_locations(features)

        pred_objectness_logits, pred_features, pred_anchor_deltas = self.rpn_head(features)
        pos_embed = []

        pred_objectness_logits = [
            # (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
            score.view(score.shape[0], -1, self.num_classes, score.shape[-2], score.shape[-1])
            .permute(0, 3, 4, 1, 2).flatten(1, -2)
            for score in pred_objectness_logits
        ]

        pred_features = [
            # (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
            feat.view(feat.shape[0], -1, 256, feat.shape[-2], feat.shape[-1])
            .permute(0, 3, 4, 1, 2).flatten(1, -2)
            for feat in pred_features
        ]

        pred_anchor_deltas = [
            # (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N, Hi*Wi*A, B)
            x.view(x.shape[0], -1, self.box_dim, x.shape[-2], x.shape[-1])
            .permute(0, 3, 4, 1, 2)
            .flatten(1, -2)
            for x in pred_anchor_deltas
        ]
        if self.training:
            # instances = [batched_input["instances"].to(self.device) for batched_input in batched_inputs]
            gt_classes, gt_shifts = self.get_ground_truth(
                images.tensor, locations, targets, pred_objectness_logits, pred_anchor_deltas)

            losses = self.losses(gt_classes, gt_shifts, pred_objectness_logits, pred_anchor_deltas)
            proposals = self.predict_proposals(
                locations, pred_objectness_logits, pred_features, pred_anchor_deltas, image_sizes)
        else:
            losses = {}
            proposals = self.simple_predict_proposals(
                locations, pred_objectness_logits, pred_features, pred_anchor_deltas, image_sizes)
        
        return proposals, losses

    def simple_predict_proposals(
        self,
        locations,
        pred_objectness_logits: List[torch.Tensor],
        pred_features: List[torch.Tensor],
        pred_anchor_deltas: List[torch.Tensor],
        image_sizes
    ):
        N = pred_objectness_logits[0].shape[0]
        proposals = []
        with torch.no_grad():
            for img_idx in range(N):
                pred_anchor_deltas_single = cat([
                    pred_anchor_delta[img_idx] for pred_anchor_delta in pred_anchor_deltas
                ])
                pred_anchor_logits_single = cat([
                    pred_anchor_logit[img_idx] for pred_anchor_logit in pred_objectness_logits
                ])
                pred_feature_single = cat([
                    pred_feature[img_idx] for pred_feature in pred_features
                ])
                locations = cat(locations)
                predicted_boxes = self.apply_deltas(
                    pred_anchor_deltas_single,
                    locations)
                scores_all = pred_anchor_logits_single.flatten()
                keep = scores_all.argsort(descending=True)
                keep = keep[:self.max_detections_per_image]

                result = Instances(image_sizes[img_idx])
                boxes_all = Boxes(predicted_boxes)
                result.proposal_boxes = boxes_all[keep]
                result.proposal_feats = pred_feature_single[keep]
                proposals.append(result)
        return proposals

    def predict_proposals(
        self,
        locations,
        pred_objectness_logits: List[torch.Tensor],
        pred_features: List[torch.Tensor],
        pred_anchor_deltas: List[torch.Tensor],
        image_sizes
    ):
        N = pred_objectness_logits[0].shape[0]
        keep_idxs = []
        proposals = []
        with torch.no_grad():
            for img_idx in range(N):
                pred_anchor_deltas_single = [
                    pred_anchor_delta[img_idx] for pred_anchor_delta in pred_anchor_deltas
                ]
                pred_anchor_logits_single = [
                    pred_anchor_logit[img_idx] for pred_anchor_logit in pred_objectness_logits
                ]
                predicted_boxes = self.apply_deltas(
                    cat(pred_anchor_deltas_single),
                    cat(locations))
                boxes_all = Boxes(predicted_boxes)
                scores_all = cat(pred_anchor_logits_single).flatten().sigmoid()
                boxes_all.clip(image_sizes[img_idx])
                keep1 = boxes_all.nonempty()
                scores_all[~keep1] = -1

                keep = scores_all.argsort(descending=True)
                keep = keep[:self.max_detections_per_image]
                keep_idxs.append(keep)
                result = Instances(image_sizes[img_idx])
                result.proposal_boxes = boxes_all[keep]
                result.objectness_logits = scores_all[keep]
                proposals.append(result)
        for img_idx in range(N):
            pred_feature_single = cat([
                pred_feature[img_idx] for pred_feature in pred_features
            ])
            pred_feature_single = pred_feature_single[keep_idxs[img_idx], :]
            proposals[img_idx].proposal_feats = pred_feature_single

        return proposals

    def losses(
            self,
            gt_labels,
            gt_boxes,
            pred_objectness_logits: List[torch.Tensor],
            pred_shift_deltas: List[torch.Tensor]):

        pred_class_logits = torch.cat(pred_objectness_logits, dim=1).view(-1, self.num_classes)
        pred_shift_deltas = torch.cat(pred_shift_deltas, dim=1).view(-1, 4)

        gt_labels = gt_labels.view(-1)
        gt_boxes = gt_boxes.view(-1, 4)

        valid_idxs = gt_labels >= 0
        # background_idxs = gt_labels == 0
        if self.num_classes == 1:
            foreground_idxs = gt_labels == 1
        else:
            foreground_idxs = (gt_labels >= 0) & (gt_labels != self.num_classes)
        num_foreground = foreground_idxs.sum()

        gt_classes_target = torch.zeros_like(pred_class_logits)
        if self.num_classes == 1:
            gt_classes_target[foreground_idxs, :] = 1
        else:
            gt_classes_target[foreground_idxs, gt_labels[foreground_idxs]] = 1

        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_foreground)
        num_foreground = torch.clamp(num_foreground / get_world_size(), min=1).item()

        loss_cls = sigmoid_focal_loss_jit(
            pred_class_logits[valid_idxs],
            gt_classes_target[valid_idxs],
            alpha=0.25,
            gamma=2.0,
            reduction="sum",
        ) / max(1.0, num_foreground)

        pred_shift_deltas = torch.cat(
            (-pred_shift_deltas[..., :2], pred_shift_deltas[..., 2:]), dim=-1)
        gt_boxes = torch.cat((-gt_boxes[..., :2], gt_boxes[..., 2:]), dim=-1)

        loss_giou = 1 - torch.diag(
            box_ops.generalized_box_iou(
                pred_shift_deltas[foreground_idxs],
                gt_boxes[foreground_idxs]))
        loss_giou = loss_giou.sum() / max(1.0, num_foreground)

        losses = {
            "loss_rpn_cls": loss_cls * self.cls_weight,
            "loss_rpn_reg": loss_giou * self.reg_weight
        }
        return losses

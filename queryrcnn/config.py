# -*- coding: utf-8 -*-
#
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.config import CfgNode as CN


def add_sparsercnn_config(cfg):
    """
    Add config for SparseRCNN.
    """
    cfg.MODEL.SparseRCNN = CN()
    cfg.MODEL.SparseRCNN.NUM_CLASSES = 80
    cfg.MODEL.SparseRCNN.NUM_PROPOSALS = 300

    # RCNN Head.
    cfg.MODEL.SparseRCNN.NHEADS = 8
    cfg.MODEL.SparseRCNN.DROPOUT = 0.0
    cfg.MODEL.SparseRCNN.DIM_FEEDFORWARD = 2048
    cfg.MODEL.SparseRCNN.ACTIVATION = 'relu'
    cfg.MODEL.SparseRCNN.HIDDEN_DIM = 256
    cfg.MODEL.SparseRCNN.NUM_CLS = 1
    cfg.MODEL.SparseRCNN.NUM_REG = 3
    cfg.MODEL.SparseRCNN.NUM_HEADS = 6

    # Dynamic Conv.
    cfg.MODEL.SparseRCNN.NUM_DYNAMIC = 2
    cfg.MODEL.SparseRCNN.DIM_DYNAMIC = 64

    # Loss.
    cfg.MODEL.SparseRCNN.CLASS_WEIGHT = 2.0
    cfg.MODEL.SparseRCNN.GIOU_WEIGHT = 2.0
    cfg.MODEL.SparseRCNN.L1_WEIGHT = 5.0
    cfg.MODEL.SparseRCNN.DEEP_SUPERVISION = True
    cfg.MODEL.SparseRCNN.NO_OBJECT_WEIGHT = 0.1

    # Focal Loss.
    cfg.MODEL.SparseRCNN.USE_FOCAL = True
    cfg.MODEL.SparseRCNN.ALPHA = 0.25
    cfg.MODEL.SparseRCNN.GAMMA = 2.0
    cfg.MODEL.SparseRCNN.PRIOR_PROB = 0.01
    cfg.MODEL.SparseRCNN.BBOX_WEIGHTS = [2.0, 2.0, 1., 1.]

    # Optimizer.
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0

    # TTA.
    cfg.TEST.AUG.MIN_SIZES = (400, 500, 600, 640, 700, 900, 1000, 1100, 1200, 1300, 1400, 1800, 800)
    cfg.TEST.AUG.CVPODS_TTA = True
    cfg.TEST.AUG.SCALE_FILTER = True
    cfg.TEST.AUG.SCALE_RANGES = ([96, 10000], [96, 10000], 
                                 [64, 10000], [64, 10000],
                                 [64, 10000], [0, 10000],
                                 [0, 10000], [0, 256],
                                 [0, 256], [0, 192],
                                 [0, 192], [0, 96],
                                 [0, 10000])
    cfg.MODEL.QueryRCNN = CN()

    cfg.MODEL.QueryRCNN.RCNNHead = 'rcnnhead'
    cfg.MODEL.QueryRCNN.WITH_POS = False

    cfg.MODEL.QueryRCNN.RPN = CN()
    cfg.MODEL.QueryRCNN.RPN.FPN_STRIDES = [8, 16, 32, 64, 128]
    cfg.MODEL.QueryRCNN.RPN.RPN_TYPE = "anchor_free"
    cfg.MODEL.QueryRCNN.RPN.NUM_CLASSES = 1
    cfg.MODEL.QueryRCNN.RPN.TOPK_INDICES = 1000
    cfg.MODEL.QueryRCNN.RPN.NUM_PROPOSALS = 100
    cfg.MODEL.QueryRCNN.RPN.OJT_ALPHA = 0.2
    cfg.MODEL.QueryRCNN.RPN.USE_FOCAL = True
    cfg.MODEL.QueryRCNN.RPN.FEAT_BRANCH = True
    cfg.MODEL.QueryRCNN.RPN.CLS_WEIGHT = 1.
    cfg.MODEL.QueryRCNN.RPN.REG_WEIGHT = 2.

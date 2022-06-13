from .qgn_head import QGN



def build_rpn_head(cfg, input_shape, in_features):
    if cfg.MODEL.QueryRCNN.RPN.RPN_TYPE == 'anchor_free':
        return QGN(cfg, input_shape, in_features)

from detectron2.config import CfgNode as CN


def add_continual_config(cfg):
    cfg.WANDB = True

    cfg.CONT = CN()
    cfg.CONT.BASE_CLS = 20
    cfg.CONT.INC_CLS = 2
    cfg.CONT.ORDER = list(range(1, 41))
    cfg.CONT.ORDER_NAME = None
    cfg.CONT.TASK = 0
    cfg.CONT.WEIGHTS = None
    cfg.CONT.MODE = "overlap"  # Choices "overlap", "disjoint", "sequential"
    cfg.CONT.INC_QUERY = False
    cfg.CONT.COSINE = False
    cfg.CONT.USE_BIAS = True
    cfg.CONT.WA_STEP = 0
    # Number of experts used in base task (task0) when MoE is enabled.
    # Incremental task t will use BASE_EXPERTS + t experts.
    cfg.CONT.BASE_EXPERTS = 2
    # SVD energy threshold for subspace update in incremental MoE training.
    # Can be overridden from script by passing: CONT.SVD_THRESHOLD <value>
    cfg.CONT.SVD_THRESHOLD = 0.95

    cfg.CONT.DIST = CN()
    cfg.CONT.DIST.POD_WEIGHT = 0.
    cfg.CONT.DIST.KD_WEIGHT = 0.
    cfg.CONT.DIST.ALPHA = 1.
    cfg.CONT.DIST.UCE = False
    cfg.CONT.DIST.UKD = False
    cfg.CONT.DIST.L2 = False
    cfg.CONT.DIST.KD_REW = False
    cfg.CONT.DIST.KD_DEEP = False
    cfg.CONT.DIST.USE_NEW = False
    cfg.CONT.DIST.EOS_POW = 0.
    cfg.CONT.DIST.CE_NEW = False
    cfg.CONT.DIST.PSEUDO = False
    cfg.CONT.DIST.PSEUDO_TYPE = 0
    cfg.CONT.DIST.IOU_THRESHOLD = 0.5
    cfg.CONT.DIST.PSEUDO_THRESHOLD = 0.
    cfg.CONT.DIST.MASK_KD = 0.
    # cfg.CONT.DIST.SANITY = 1.
    # cfg.CONT.DIST.WEIGHT_MASK = 1.
    
    # Parameters for ECLIPSE
    cfg.CONT.SOFTCLS = True
    cfg.CONT.BACKBONE_FREEZE = False
    cfg.CONT.CLS_HEAD_FREEZE = False
    cfg.CONT.MASK_HEAD_FREEZE = False
    cfg.CONT.PIXEL_DECODER_FREEZE = False
    cfg.CONT.QUERY_EMBED_FREEZE = False
    cfg.CONT.TRANS_DECODER_FREEZE = False
    cfg.CONT.LOGIT_MANI_DELTAS = None
    ###########################################
    cfg.CONT.QUERY_FEAT = False

    cfg.CONT.THRESHOLD = 0.5
    # SVD energy ratio for old-router subspace projection.
    # Used in MoE new_router.weight orthogonal gradient projection.
    cfg.CONT.ROUTER_SVD_ENERGY = 0.99
    #################################

    cfg.DATASETS.DATASET_NEED_MAP = [False, ]
    # dataset type, selected from ['video_instance', 'video_panoptic', 'video_semantic',
    #                              'image_instance', 'image_panoptic', 'image_semantic']
    cfg.DATASETS.DATASET_TYPE = ['video_instance', ]
    cfg.DATASETS.DATASET_TYPE_TEST = ['video_instance', ]
    cfg.MODEL.MASK_FORMER.TEST.TASK = 'vis' ###############################################################
    cfg.INPUT.REVERSE_AGU = False

    ####################################
    

"""
Author: Isabella Liu 7/18/21
Feature: Config file for messy-table-dataset
"""

from yacs.config import CfgNode as CN

_C = CN()
cfg = _C

# Directories
_C.DIR = CN()
_C.DIR.DATASET = '/code/dataset_local_v9/training'  #  directory of your training dataset

# Split files
_C.SPLIT = CN()
_C.SPLIT.TRAIN = '/code/dataset_local_v9/training_lists/all_train.txt'  # training lists of your training dataset
_C.SPLIT.VAL = '/code/dataset_local_v9/training_lists/all_val.txt'  # training lists of your validation dataset
_C.SPLIT.OBJ_NUM = 18  # Note: table + ground - 17th

_C.SPLIT.LEFT = '0128_irL_denoised_half.png'
_C.SPLIT.RIGHT = '0128_irR_denoised_half.png'
_C.SPLIT.DEPTHL = 'depthL.png'
_C.SPLIT.DEPTHR = 'depthR.png'
_C.SPLIT.META = 'meta.pkl'
_C.SPLIT.LABEL = 'irL_label_image.png'

_C.SPLIT.SIM_REALSENSE = '0128_depth_denoised.png'
_C.SPLIT.REAL_REALSENSE = '1024_depth_real.png'

# Configuration for testing on real dataset
_C.REAL = CN()
_C.REAL.DATASET = '/code/real_dataset_local_v9'  # path to your real testing dataset
_C.REAL.DEPTHPATH = '/cephfs/datasets/iccv_pnp/messy-table-dataset/real_v9/training'
_C.REAL.TRAIN = '/cephfs/jianyu/newTrain.txt'
_C.REAL.LEFT = '1024_irL_real_1080.png'
_C.REAL.RIGHT = '1024_irR_real_1080.png'
_C.REAL.PAD_WIDTH = 960
_C.REAL.PAD_HEIGHT = 544

# Solver args
_C.SOLVER = CN()
_C.SOLVER.LR_CASCADE = 0.001        # base learning rate for cascade
_C.SOLVER.LR_G = 0.0002             # base learning rate for G
_C.SOLVER.LR_D = 0.0003             # base learning rate for D
_C.SOLVER.LR_EPOCHS = '5,10,15:2'   # the epochs to decay lr: the downscale rate
_C.SOLVER.LR_STEPS = '5,10,15:2'    # the steps to decay lr: the downscale rate
_C.SOLVER.EPOCHS = 20               # number of epochs to train
_C.SOLVER.STEPS = 10000             # number of steps to train
_C.SOLVER.BATCH_SIZE = 1            # batch size
_C.SOLVER.TEST_BATCH_SIZE = 1
_C.SOLVER.NUM_WORKER = 1            # num_worker in dataloader
# Type of optimizer
_C.SOLVER.TYPE = "RMSprop"

# Basic parameters of solvers
# Notice to change learning rate according to batch size
_C.SOLVER.BASE_LR = 0.001

_C.SOLVER.WEIGHT_DECAY = 0.0

# Specific parameters of solvers
_C.SOLVER.RMSprop = CN()
_C.SOLVER.RMSprop.alpha = 0.9

_C.SOLVER.SGD = CN()
_C.SOLVER.SGD.momentum = 0.9

# ---------------------------------------------------------------------------- #
# Scheduler (learning rate schedule)
# ---------------------------------------------------------------------------- #
_C.SCHEDULER = CN()
_C.SCHEDULER.TYPE = ""

_C.SCHEDULER.INIT_EPOCH = 2
_C.SCHEDULER.GT_OCC_EPOCH = -1
_C.SCHEDULER.MAX_EPOCH = 2

_C.SCHEDULER.StepLR = CN()
_C.SCHEDULER.StepLR.step_size = 0
_C.SCHEDULER.StepLR.gamma = 0.1

_C.SCHEDULER.MultiStepLR = CN()
_C.SCHEDULER.MultiStepLR.milestones = ()
_C.SCHEDULER.MultiStepLR.gamma = 0.1

# Model args
_C.ARGS = CN()
_C.ARGS.MAX_DISP = 192              # maximum disparity
_C.ARGS.MODEL = 'gwcnet-c'
_C.ARGS.GRAD_METHOD = 'detach'
_C.ARGS.NDISP = (48, 24)            # ndisps
_C.ARGS.DISP_INTER_R = (4, 1)       # disp_intervals_ratio
_C.ARGS.DLOSSW = (0.5, 2.0)         # depth loss weight for different stage
_C.ARGS.CR_BASE_CHS = (32, 32, 16)  # cost regularization base channels
_C.ARGS.USING_NS = True             # using neighbor search
_C.ARGS.NS_SIZE = 3                 # nb_size
_C.ARGS.CROP_HEIGHT = 256           # crop height
_C.ARGS.CROP_WIDTH = 512            # crop width
_C.ARGS.TEST_CROP_HEIGHT = 540           # crop height
_C.ARGS.TEST_CROP_WIDTH = 960            # crop width

# Data Augmentation
_C.DATA_AUG = CN()
_C.DATA_AUG.BRIGHT_MIN = 0.4
_C.DATA_AUG.BRIGHT_MAX = 1.4
_C.DATA_AUG.CONTRAST_MIN = 0.8
_C.DATA_AUG.CONTRAST_MAX = 1.2
_C.DATA_AUG.GAUSSIAN_MIN = 0.1
_C.DATA_AUG.GAUSSIAN_MAX = 2
_C.DATA_AUG.GAUSSIAN_KERNEL = 9

# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------

_C.MODEL = CN()
_C.MODEL.WEIGHT = ""

_C.MODEL.BASE_CHANNEL = 32
_C.MODEL.MAX_DISP = 136

# loss type: ["SELF_SUPERVISE", "SUPERVISE"]
_C.MODEL.LOSS_TYPE = "SELF_SUPERVISE"
_C.MODEL.INVALID_REG_WEIGHT = 1.0
_C.MODEL.INVALID_WEIGHT = 1.0
_C.MODEL.INVALID_THRESHOLD = (0.5, 0.8, )

_C.MODEL.SELF_SUPERVISE = CN()
_C.MODEL.SELF_SUPERVISE.LCN_KERNEL_SIZE = 9
_C.MODEL.SELF_SUPERVISE.WINDOW_SIZE = 33
_C.MODEL.SELF_SUPERVISE.SIGMA_WEIGHT = 2

_C.TRAIN = CN()

_C.TRAIN.BATCH_SIZE = 1

# The period to save a checkpoint
_C.TRAIN.CHECKPOINT_PERIOD = 1000
_C.TRAIN.LOG_PERIOD = 50
_C.TRAIN.FILE_LOG_PERIOD = 100000
# The period to validate
_C.TRAIN.VAL_PERIOD = 0
# Data augmentation. The format is "method" or ("method", *args)
_C.TRAIN.AUGMENTATION = ()

# Regex patterns of modules and/or parameters to freeze
# For example, ("bn",) will freeze all batch normalization layers" weight and bias;
# And ("module:bn",) will freeze all batch normalization layers" running mean and var.
_C.TRAIN.FROZEN_PATTERNS = ()

_C.TRAIN.VAL_METRIC = "<1_cor"
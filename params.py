from pathlib import Path


class Params:

    class MoCo:
        # MoCo Arch
        DIM = 128
        K = 4096
        M = 0.99
        T = 0.07
        ADD_MLP_HEAD = False

        # MoCo Training
        EPOCHS = 200
        LR = 0.03
        WEIGHT_DECAY = 5e-4
        MOMENTUM = 0.9
        BATCH_SIZE = 128

    class KNN:
        # K-nearest-neighbor
        K = 200
        T = 0.1

    class Classifier:
        # Transfer Learning
        LR = 0.03
        MOMENTUM = 0.9
        WEIGHT_DECAY = 0
        BATCH_SIZE = 128
        EPOCHS = 50

    INPUT_SIZE = 128
    WORKING_DIR = Path.cwd()
    RESULTS_DIR = WORKING_DIR
    SSL_CKPT_PATH = WORKING_DIR / 'pl_logs_moco/version_3/checkpoints/epoch=87-step=12935.ckpt'

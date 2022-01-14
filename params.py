from pathlib import Path


class Params:

    class MoCo:
        # MoCo Arch
        DIM = 128
        K = 4096
        M = 0.99
        T = 0.1
        ARCH = 'resnet18'
        BN_SPLITS = 1
        SYMMETRIC = True

        # MoCo Training
        EPOCHS = 10
        LR = 0.06
        WEIGHT_DECAY = 5e-4
        MOMENTUM = 0.9
        BATCH_SIZE = 32

    class KNN:
        # K-nearest-neighbor
        K = 200
        T = 0.1

    RESULTS_DIR = '.'

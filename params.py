from pathlib import Path


class Params:

    class MoCo:
        # MoCo Arch
        DIM = 128
        K = 4096
        M = 0.99
        T = 0.07
        SYMMETRIC = True

        # MoCo Training
        EPOCHS = 500
        LR = 0.06
        WEIGHT_DECAY = 5e-4
        MOMENTUM = 0.9
        BATCH_SIZE = 32

    class KNN:
        # K-nearest-neighbor
        K = 200
        T = 0.1

    class Classifier:
        # Transfer Learning
        LR = 1e-3
        MOMENTUM = 0.9
        WEIGHT_DECAY = 5e-4
        BATCH_SIZE = 128
        EPOCHS = 50

    RESULTS_DIR = '.'

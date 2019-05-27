class StitchOptions(object):
    BASE_ERROR_RATE = 0.0
    label_decoder = {1: 'A', 2: 'C', 3: 'G', 4: 'T', 0: ''}
    MATCH_PENALTY = 4
    MISMATCH_PENALTY = 6
    GAP_PENALTY = 8
    GAP_EXTEND_PENALTY = 2
    MIN_SEQUENCE_REQUIRED_FOR_MULTITHREADING = 2
    OVERLAP_THRESHOLD = 8
    KMER_SIZE = 15


class ImageSizeOptions(object):
    IMAGE_HEIGHT = 90
    IMAGE_CHANNELS = 1
    SEQ_LENGTH = 1000
    SEQ_OVERLAP = 200
    LABEL_LENGTH = SEQ_LENGTH

    TOTAL_BASE_LABELS = 5
    TOTAL_RLE_LABELS = 11


class TrainOptions(object):
    TRAIN_WINDOW = 100
    WINDOW_JUMP = 50
    GRU_LAYERS = 1
    HIDDEN_SIZE = 128
    CLASS_WEIGHTS = [0.3, 0.5, 0.5, 0.5, 0.5, 0.8, 0.9, 1.0, 1.0, 1.0, 0.9]


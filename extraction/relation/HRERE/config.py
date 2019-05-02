# ----------------------- PATH ------------------------

ROOT_PATH = "."
DATA_PATH = "%s/data" % ROOT_PATH
WN18_DATA_PATH = "%s/wn18" % DATA_PATH
WN18RR_DATA_PATH = "%s/wn18rr" % DATA_PATH
FB15K_DATA_PATH = "%s/fb15k" % DATA_PATH
FB15K237_DATA_PATH = "%s/fb15k-237" % DATA_PATH
FB3M_DATA_PATH = "%s/fb3m" % DATA_PATH

LOG_PATH = "%s/log" % ROOT_PATH
CHECKPOINT_PATH = "%s/checkpoint" % ROOT_PATH

# ----------------------- DATA ------------------------

DATASET = {}

WN18_TRAIN_RAW = "%s/train.txt" % WN18_DATA_PATH
WN18_VALID_RAW = "%s/valid.txt" % WN18_DATA_PATH
WN18_TEST_RAW = "%s/test.txt" % WN18_DATA_PATH
WN18_TRAIN = "%s/digitized_train.txt" % WN18_DATA_PATH
WN18_VALID = "%s/digitized_valid.txt" % WN18_DATA_PATH
WN18_TEST = "%s/digitized_test.txt" % WN18_DATA_PATH
WN18_E2ID = "%s/e2id.txt" % WN18_DATA_PATH
WN18_R2ID = "%s/r2id.txt" % WN18_DATA_PATH

DATASET["wn18"] = {
    "train_raw": WN18_TRAIN_RAW,
    "valid_raw": WN18_VALID_RAW,
    "test_raw": WN18_TEST_RAW,
    "train": WN18_TRAIN,
    "valid": WN18_VALID,
    "test": WN18_TEST,
    "e2id": WN18_E2ID,
    "r2id": WN18_R2ID,
}

WN18RR_TRAIN_RAW = "%s/train.txt" % WN18RR_DATA_PATH
WN18RR_VALID_RAW = "%s/valid.txt" % WN18RR_DATA_PATH
WN18RR_TEST_RAW = "%s/test.txt" % WN18RR_DATA_PATH
WN18RR_TRAIN = "%s/digitized_train.txt" % WN18RR_DATA_PATH
WN18RR_VALID = "%s/digitized_valid.txt" % WN18RR_DATA_PATH
WN18RR_TEST = "%s/digitized_test.txt" % WN18RR_DATA_PATH
WN18RR_E2ID = "%s/e2id.txt" % WN18RR_DATA_PATH
WN18RR_R2ID = "%s/r2id.txt" % WN18RR_DATA_PATH

DATASET["wn18rr"] = {
    "train_raw": WN18RR_TRAIN_RAW,
    "valid_raw": WN18RR_VALID_RAW,
    "test_raw": WN18RR_TEST_RAW,
    "train": WN18RR_TRAIN,
    "valid": WN18RR_VALID,
    "test": WN18RR_TEST,
    "e2id": WN18RR_E2ID,
    "r2id": WN18RR_R2ID,
}

FB15K_TRAIN_RAW = "%s/train.txt" % FB15K_DATA_PATH
FB15K_VALID_RAW = "%s/valid.txt" % FB15K_DATA_PATH
FB15K_TEST_RAW = "%s/test.txt" % FB15K_DATA_PATH
FB15K_TRAIN = "%s/digitized_train.txt" % FB15K_DATA_PATH
FB15K_VALID = "%s/digitized_valid.txt" % FB15K_DATA_PATH
FB15K_TEST = "%s/digitized_test.txt" % FB15K_DATA_PATH
FB15K_E2ID = "%s/e2id.txt" % FB15K_DATA_PATH
FB15K_R2ID = "%s/r2id.txt" % FB15K_DATA_PATH

DATASET["fb15k"] = {
    "train_raw": FB15K_TRAIN_RAW,
    "valid_raw": FB15K_VALID_RAW,
    "test_raw": FB15K_TEST_RAW,
    "train": FB15K_TRAIN,
    "valid": FB15K_VALID,
    "test": FB15K_TEST,
    "e2id": FB15K_E2ID,
    "r2id": FB15K_R2ID,
}

FB15K237_TRAIN_RAW = "%s/train.txt" % FB15K237_DATA_PATH
FB15K237_VALID_RAW = "%s/valid.txt" % FB15K237_DATA_PATH
FB15K237_TEST_RAW = "%s/test.txt" % FB15K237_DATA_PATH
FB15K237_TRAIN = "%s/digitized_train.txt" % FB15K237_DATA_PATH
FB15K237_VALID = "%s/digitized_valid.txt" % FB15K237_DATA_PATH
FB15K237_TEST = "%s/digitized_test.txt" % FB15K237_DATA_PATH
FB15K237_E2ID = "%s/e2id.txt" % FB15K237_DATA_PATH
FB15K237_R2ID = "%s/r2id.txt" % FB15K237_DATA_PATH

DATASET["fb15k237"] = {
    "train_raw": FB15K237_TRAIN_RAW,
    "valid_raw": FB15K237_VALID_RAW,
    "test_raw": FB15K237_TEST_RAW,
    "train": FB15K237_TRAIN,
    "valid": FB15K237_VALID,
    "test": FB15K237_TEST,
    "e2id": FB15K237_E2ID,
    "r2id": FB15K237_R2ID,
}

FB3M_TRAIN_RAW = "%s/train.txt" % FB3M_DATA_PATH
FB3M_VALID_RAW = "%s/valid.txt" % FB3M_DATA_PATH
FB3M_TEST_RAW = "%s/test.txt" % FB3M_DATA_PATH
FB3M_TRAIN = "%s/digitized_train.txt" % FB3M_DATA_PATH
FB3M_VALID = "%s/digitized_valid.txt" % FB3M_DATA_PATH
FB3M_TEST = "%s/digitized_test.txt" % FB3M_DATA_PATH
FB3M_E2ID = "%s/e2id.txt" % FB3M_DATA_PATH
FB3M_R2ID = "%s/r2id.txt" % FB3M_DATA_PATH

DATASET["fb3m"] = {
    "train_raw": FB3M_TRAIN_RAW,
    "valid_raw": FB3M_VALID_RAW,
    "test_raw": FB3M_TEST_RAW,
    "train": FB3M_TRAIN,
    "valid": FB3M_VALID,
    "test": FB3M_TEST,
    "e2id": FB3M_E2ID,
    "r2id": FB3M_R2ID,
}

# ----------------------- PARAM -----------------------

RANDOM_SEED = None

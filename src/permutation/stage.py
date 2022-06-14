from enum import Enum, auto


class Stage(Enum):
    TRAIN = "train"
    TEST = "test"
    VAL = "val"
    PERM = "perm"

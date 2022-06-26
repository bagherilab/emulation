from enum import Enum, auto


class Stage(Enum):
    TRAIN = "train"
    TEST = "test"
    VAL = "val"
    PERM = "perm"


class IncorrectStageException(Exception):
    def __init__(self, correct_stage: Stage):
        self.message = f"Incorrect stage for {correct_stage.name} method."
        super().__init__(self.message)

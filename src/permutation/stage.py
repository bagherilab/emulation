from enum import Enum


class Stage(Enum):
    """todo"""

    TRAIN = "train"
    TEST = "test"
    VAL = "val"
    PERM = "perm"


class IncorrectStageException(Exception):
    """todo"""

    def __init__(self, correct_stage: Stage):
        """todo"""
        self.message = f"Incorrect stage for {correct_stage.name} method."
        super().__init__(self.message)

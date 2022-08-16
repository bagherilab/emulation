from enum import Enum


class Stage(Enum):
    """
    Enum class for specifying the stage of the experiment
    is required for sequential training, testing, etc.
    """

    TRAIN = "train"
    TEST = "test"
    VAL = "val"
    PERM = "perm"


class IncorrectStageException(Exception):
    """Exception for stage inconsistencies between classes. Check against expected enum."""

    def __init__(self, incorrect_stage: Stage, correct_stage: Stage) -> None:
        """Raise error message with correct stage specified in declaration."""
        self.message = (
            f"Incorrect stage passed ({incorrect_stage.name}) for {correct_stage.name} method."
        )
        super().__init__(self.message)

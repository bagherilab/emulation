from pathlib import Path

from runner import Runner


class ExperimentTracker:
    def __init__(
        self, log_dir: str, runner: Runner, hparam_set: Optional[list[hparams]] = None
    ):
        pass

    @staticmethod
    def _validate_log_dir(log_dir: str, create: bool = True):
        log_path = Path(log_dir).resolve()
        if log_path.exists():
            return
        elif not log_path.exists() and create:
            log_path.mkdir(parents=True)
        else:
            raise NotADirectoryError(f"log_dir {log_dir} does not exist.")

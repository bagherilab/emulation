from pathlib import Path
import glob


def validate_dir(dir_string: str, create: bool = True) -> None:
    """
    validates the directory exists and creates if not,
    or raises an error if create is false
    """
    log_path = Path(dir_string).resolve()
    if log_path.exists():
        return
    if not log_path.exists() and create:
        log_path.mkdir(parents=True)
    else:
        raise NotADirectoryError(f"dir {dir_string} does not exist.")


def clean_dir(dir_string: str) -> None:
    """todo"""
    files = glob.glob("dir_string/**/*.csv", recursive=True)

    for f in files:
        try:
            os.remove(f)
        except OSError as e:
            print("Error: %s : %s" % (f, e.strerror))

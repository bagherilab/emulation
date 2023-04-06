import os
from pathlib import Path
import glob


def validate_dir(dir_string: str, create: bool = True) -> None:
    """
    Validates the directory exists and creates if not

    Raises
    ------
    NotADirectoryError:
        Raises error if directory doesn't exist and create is False
    """
    log_path = Path(dir_string).resolve()
    if log_path.exists():
        return
    if not log_path.exists() and create:
        log_path.mkdir(parents=True)
    else:
        raise NotADirectoryError(f"dir {dir_string} does not exist.")


def clean_dir(dir_string: str) -> None:
    """
    Removes all of the CSV files from a directory

    Parameters
    ----------
    dir_string :
        String representing the directory to clean
    """
    files = glob.glob("dir_string/**/*.csv", recursive=True)

    for file in files:
        try:
            os.remove(file)
        except OSError as exc:
            print("Error: %s : %s" % (file, exc.strerror))

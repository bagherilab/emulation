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

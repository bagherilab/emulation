from pathlib import Path
from abc import ABC, abstractmethod

import pandas as pd


class Loader(ABC):
    path: str

    @abstractmethod
    def load(path: str | Path) -> pd.DataFrame:
        ...


class CSVLoader(Loader):
    path: str

    def load(self) -> pd.DataFrame:
        return pd.read_csv(path)

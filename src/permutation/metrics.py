from dataclasses import dataclass, field
from abc import ABC, abstractmethod

import pandas as pd

from permutation.stage import Stage


class Metric(ABC):  # pylint: disable=too-few-public-methods
    """
    Abstract class for implementing different metrics,
    specifically with methods to ensure correct csv handling by Logger object
    through pandas

    Abstract Methods
    ----------------
    to_pandas() -> pd.DataFrame:
        Organize data in metric class into tabular format, and return a pandas Dataframe
    """

    name: str
    stage: Stage

    @abstractmethod
    def to_pandas(self) -> pd.DataFrame:
        """Implement a method to export to dataframe"""


@dataclass
class BatchMetric(Metric):
    """
    Class for holding aggregated results from model performance evaluation

    Attributes
    ----------
    name :
        Name of collection of stored data
    values :
        List of values/data
    total :
        Sum of values
    average :
        Average of values
    length :
        Number of values contained
    stage:
        Stage of model training process data is associated with

    Methods
    -------
    update(new_value):
        Adds a value to the list

    batchupdate(new_values):
        Adds iterable of values to list

    set_stage(stage):
        Updates the associated stage of the metric, e.g. Stage.VAL, Stage.PERM
    """

    name: str
    value_type: str
    stage: Stage
    values: list[float] = field(default_factory=list)
    total: float = 0.0
    average: float = 0.0
    length: int = 0

    def update(self, new_value: float) -> None:
        """
        Method to update `values` attribute with `new_value`

        Parameters
        ----------
        new_value : 
            Value to add to data container

        """
        self.values.append(new_value)
        self.total += new_value
        self.length += 1
        self.average = self.total / self.length

    def batchupdate(self, new_values: list[float]) -> None:
        """
        Method to update `values` attribute with list of values, `new_values`

        Parameters
        ----------
        new_values : 
            List of values to add to data container

        """
        for value in new_values:
            self.update(value)

    def to_pandas(self) -> pd.DataFrame:
        """Method to export to dataframe"""
        return pd.DataFrame(self.values, columns=[self.value_type])


@dataclass
class SequentialMetric(Metric):
    """
    Class for holding linked sequential results from model performance evaluation
    (e.g. training metrics associated with size of training data)

    Attributes
    ----------
    name :
        Name of collection of stored data
    values :
        List of values/data
    nums :
        Ordered int values associated with each value by index
    stage:
        Stage of model training process data is associated with, e.g. Stage.TRAIN, Stage.TEST

    Methods
    -------
    update(new_value, n):
        Adds a value to the list

    batchupdate(new_values, new_ns):
        Adds iterable of values to list

    """

    name: str
    value_type: str
    stage: Stage
    values: list[float] = field(default_factory=list)
    nums: list[int] = field(default_factory=list)

    def update(self, new_value: float, n: int) -> None:
        """Method to update `values` and `nums` attribute with `new_value` and `n`"""
        self._checkorder(n)
        self.values.append(new_value)
        self.nums.append(n)

    def batchupdate(self, new_values: list[float], new_ns: list[int]) -> None:
        """Add list of floats and ints to `values` and `nums`"""
        for value, num in zip(new_values, new_ns):
            self.update(value, num)

    def _checkorder(self, n: int) -> None:
        """ensure nums remains in sequential order"""
        if self.nums and n <= self.nums[-1]:
            raise ValueError("New n is smaller than largest value in sequential list.")

    def to_pandas(self) -> pd.DataFrame:
        """Method to export to dataframe"""
        return pd.DataFrame(list(zip(self.nums, self.values)), columns=["nums", self.value_type])

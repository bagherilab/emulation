from typing import Optional, Iterator
from dataclasses import dataclass, field

from permutation.stage import Stage


@dataclass
class BatchMetric:
    """
    Class for holding aggregated results from model performance evaluation

    Attributes
    ----------
    name :
        name of collection of stored data
    values :
        list of values/data
    total :
        sum of values
    average :
        average of values
    length :
        number of values contained
    stage:
        stage of model training process data is associated with

    Methods
    -------
    update(new_value):
        adds a value to the list

    batchupdate(new_values):
        adds iterable of values to list

    set_stage(stage):
        updates the associated stage of the metric, e.g. Stage.VAL, Stage.PERM
    """

    name: str

    values: list[float] = field(default_factory=list)
    total: float = 0.0
    average: float = 0.0
    length: int = 0
    stage: Optional[Stage] = None

    def update(self, new_value: float) -> None:
        """
        method to update `values` attribute with `new_value`

        Parameters
        ----------
        new_value: value to add to data container

        """
        self.values.append(new_value)
        self.total += new_value
        self.length += 1
        self.average = self.total / self.length

    def batchupdate(self, new_values: list[float]) -> None:
        """
        method to update `values` attribute with list of values, `new_values`

        Parameters
        ----------
        new_values: list of values to add to data container

        """
        for value in new_values:
            self.update(value)

    def set_stage(self, stage: Stage) -> None:
        """
        method to update `stage` attribute with Stage variable, e.g. Stage.TEST

        Parameters
        ----------
        stage: stage variable

        """
        self.stage = stage


@dataclass
class SequentialMetric:
    """
    Class for holding linked sequential results from model performance evaluation
    (e.g. training metrics associated with size of training data)

    Attributes
    ----------
    name :
        name of collection of stored data
    values :
        list of values/data
    nums :
        ordered int values associated with each value by index
    stage:
        stage of model training process data is associated with, e.g. Stage.TRAIN, Stage.TEST

    Methods
    -------
    update(new_value, n):
        adds a value to the list

    batchupdate(new_values, new_ns):
        adds iterable of values to list


    """

    name: str
    values: list[float] = field(default_factory=list)
    nums: list[int] = field(default_factory=list)
    stage: Optional[Stage] = None

    def update(self, new_value: float, n: int) -> None:
        """todo"""
        self._checkorder(n)
        self.values.append(new_value)
        self.nums.append(n)

    def batchupdate(self, new_values: list[float], new_ns: list[int]) -> None:
        """todo"""
        for value, num in zip(new_values, new_ns):
            self.update(value, num)

    def set_stage(self, stage: Stage) -> None:
        """
        method to update `stage` attribute with Stage variable, e.g. Stage.TEST

        Parameters
        ----------
        stage: stage variable

        """
        self.stage = stage

    def zipped(self) -> Iterator[tuple[float, int]]:
        """todo"""
        return zip(self.values, self.nums)

    def _checkorder(self, n: int) -> None:
        """ensure nums remains in sequential order"""
        if self.nums and n <= self.nums[-1]:
            raise ValueError("New n is smaller than largest value in sequential list.")

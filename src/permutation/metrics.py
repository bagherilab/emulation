from collections.abc import Iterator
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


@dataclass
class Metric(ABC):
    @abstractmethod
    def update(arg):
        ...

    @abstractmethod
    def batchupdate(arg):
        ...


@dataclass
class BatchMetric(Metric):
    name: str
    values: list[float] = field(default_factory=list)
    total: float = 0.0
    average: float = 0.0
    length: int = 0

    def update(self, value: float) -> None:
        self.values.append(value)
        self.total += value
        self.length += 1
        self.average = self.total / self.length

    def batchupdate(self, new_values: list[float]):
        for value in new_values:
            self.update(value)


@dataclass
class SequentialMetric(Metric):
    name: str
    values: list[float] = field(default_factory=list)
    nums: list[int] = field(default_factory=list)

    def update(self, value: float, n: int) -> None:
        self._checkorder(n)
        self.values.append(value)
        self.nums.append(n)

    def batchupdate(self, new_values: list[float], new_n: list[int]) -> None:
        for value, num in zip(new_values, new_n):
            self.update(value, num)

    def zipped(self) -> Iterator[tuple[float, int]]:
        return zip(self.values, self.nums)

    def _checkorder(self, n: int) -> None:
        if self.nums and n <= self.nums[-1]:
            raise ValueError("New n is smaller than largest value in sequential list.")

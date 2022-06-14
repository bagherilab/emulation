from typing import Generator
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
    n: list[int] = field(default_factory=list)

    def update(self, value: float, n: int) -> None:
        self._checkorder(n)
        self.values.append(value)
        self.n.append(n)

    def batchupdate(self, new_values: list[float], new_n: list[n]):
        for value, n in zip(new_values, new_n):
            self.update(value, n)

    def zipped(self) -> Iterator[tuple[float, int]]:
        return zip(self.values, self.n)

    def _checkorder(self, n):
        if self.n and n <= self.n[-1]:
            raise ValueError("New n is smaller than largest value in sequential list.")

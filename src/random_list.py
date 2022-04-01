from random import Random
from typing import TypeVar, Generic, List, Final

T = TypeVar('T')


class WeightedEntry(Generic[T]):

    def __init__(self, weight: float, data: T):
        self.weight: Final = weight
        self.data: Final = data


class RandomList(Generic[T]):

    def __init__(self):
        self._list: List[WeightedEntry[T]] = []
        self._sum_weight = 0

    def add_entry(self, weight: float, obj: T):
        if weight <= 0:
            raise AssertionError("invalid weight: " + str(weight))
        self._list.append(WeightedEntry(weight, obj))
        self._sum_weight += weight

    def get_random_entry(self, rand_number: float):
        if rand_number < 0 or rand_number > 1 or len(self._list) == 0 or self._sum_weight <= 0:
            raise AssertionError("invalid list")
        rand_number *= self._sum_weight
        for entry in self._list:
            rand_number -= entry.weight
            if rand_number < 0:
                return entry.data
        raise AssertionError("failed to find entry")

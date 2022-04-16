import numpy as np
import math
from random_list import RandomList


class GaussianVariable(object):

    def __init__(self, n: int, mean: np.ndarray, a: np.ndarray):
        self.n = n
        self.mean = mean
        self.a = a
        if mean.shape != (n,):
            raise AssertionError("mean dimension mismatch")
        if a.shape != (n, n):
            raise AssertionError("cov dimension mismatch")

    def sample_data(self):
        iid = np.random.standard_normal((self.n,))
        return self.mean + np.matmul(self.a, iid)

    def get_density(self, data: np.ndarray):
        c = np.matmul(self.a, np.transpose(self.a))
        x = data - self.mean
        coef = math.pow(2 * math.pi * np.linalg.det(c), -1 / 2)
        expm = np.matmul(np.matmul(x, np.linalg.inv(c)), np.transpose(x)) / 2
        return coef * math.exp(expm)


class GaussianMixture(RandomList[GaussianVariable]):

    def __init__(self, n: int):
        super().__init__()
        self.n = n

    def add_gaussian(self, weight: float, mean: np.ndarray, a: np.ndarray):
        self.add_entry(weight, GaussianVariable(self.n, mean, a))

    def get_random_data(self, size: int, seed: int = 0):
        np.random.seed(seed)
        choice = np.random.random((size,))
        return np.asarray([self.get_random_entry(val).sample_data() for val in choice], np.float)

    def get_density(self, data: np.ndarray):
        chance = 0
        for entry in super()._list:
            chance += entry.weight * entry.data.get_density(data)
        return chance

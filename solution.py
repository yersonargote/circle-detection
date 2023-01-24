from dataclasses import dataclass

import numpy as np


@dataclass
class Solution:
    cells: np.ndarray
    fitness: float

    def __lt__(self, other):
        return self.fitness < other.fitness

    def __ge__(self, other):
        return self.fitness > other.fitness

    def __eq__(self, other):
        return self.fitness == other.fitness

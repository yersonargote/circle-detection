from dataclasses import dataclass

import numpy as np


@dataclass
class Solution:
    cells: np.ndarray
    fitness: float

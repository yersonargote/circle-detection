from copy import deepcopy
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from problem import CircleDetection
from solution import Solution


@dataclass
class GHS:
    problem: CircleDetection
    memory: np.ndarray
    max_iterations: int
    N: int
    HMCR: float
    PAR: float
    BW: float
    BWmin: float
    BWmax: float

    def init_harmony(self):
        cells = self.problem.circle()
        fitness = self.problem.evaluate(cells)
        return Solution(cells=cells, fitness=fitness)

    def update_memory(self, solution):
        if solution.fitness > self.memory[-1].fitness:
            self.memory[-1] = solution
            self.memory = np.array(sorted(self.memory, reverse=False))

    def update_bw(self, k: int):
        if k < 2 / 3 * self.max_iterations:
            self.BW = self.BWmax - (self.BWmax - self.BWmin) / (2 * self.max_iterations)
        else:
            self.BW = self.BWmin

    def solve(self) -> Tuple:
        self.memory = np.array([self.init_harmony() for _ in range(self.N)])
        self.memory = np.array(sorted(self.memory, reverse=False))
        best: Solution = self.memory[0]
        solutions = []
        for i in range(self.max_iterations):
            cells = np.zeros(self.problem.size)
            for j in range(self.problem.size):
                rnd = np.random.uniform()
                if rnd < self.HMCR:
                    x = np.random.randint(self.N)
                    cells[j] = self.memory[x].cells[j]
                    rnd = np.random.uniform()
                    if rnd < self.PAR:
                        rnd = np.random.uniform()
                        cells[j] = np.around(self.memory[0].cells[j] + rnd * self.BW)
                    self.update_bw(i)
                else:
                    rnd = np.random.uniform()
                    cells[j] = 1 + np.around(rnd * self.problem.edges.shape[j % 2])
            fitness = self.problem.evaluate(cells=cells)
            solution = Solution(cells=cells, fitness=fitness)
            self.update_memory(solution)
            if self.memory[0] < best:
                best = deepcopy(self.memory[0])
            if np.isclose(best.fitness, self.problem.optimal):
                solutions.append(best)
                return np.array(solutions), best
            solutions.append(best.fitness)
        return np.array(solutions), best

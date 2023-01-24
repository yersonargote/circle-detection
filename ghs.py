from dataclasses import dataclass

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

    def init_harmony(self):
        cells = self.problem.circle()
        fitness = self.problem.evaluate(cells)
        return Solution(cells=cells, fitness=fitness)

    def update_memory(self, solution):
        if solution.fitness > self.memory[-1].fitness:
            self.memory[-1] = solution
            self.memory = np.array(sorted(self.memory, reverse=False))

    def solve(self) -> Solution:
        self.memory = np.array([self.init_harmony() for _ in range(self.N)])
        self.memory = np.array(sorted(self.memory, reverse=False))
        best: Solution = self.memory[0]
        for _ in range(self.max_iterations):
            cells = np.zeros(self.problem.size)
            for j in range(self.problem.size):
                rnd = np.random.uniform()
                if rnd <= self.HMCR:
                    x = np.random.randint(self.N)
                    cells[j] = self.memory[x].cells[j]
                    rnd = np.random.uniform()
                    if rnd <= self.PAR:
                        cells[j] = self.memory[0].cells[j]
                else:
                    cells = self.problem.circle()
            fitness = self.problem.evaluate(cells=cells)
            solution = Solution(cells=cells, fitness=fitness)
            self.update_memory(solution)
            if self.memory[0] < best:
                best = self.memory[0]
            if best.fitness == self.problem.optimal:
                return best
        return self.memory[0]

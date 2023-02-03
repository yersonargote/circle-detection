from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from problem import CircleDetection
from solution import Solution


@dataclass
class GWO:
    """Grey Wolf Optimizer"""

    max_iterations: int
    N: int
    problem: CircleDetection
    population: np.ndarray
    a: float
    alpha: Solution
    beta: Solution
    delta: Solution
    convergence: np.ndarray

    # def init_population(self):
    #     it = 0
    #     while it < self.N:
    #         wolf: Solution = self.init_wolf()
    #         if wolf.fitness > 1:
    #             continue
    #         self.population[it] = wolf
    #         it += 1

    def init_population(self):
        for i in range(self.N):
            self.population[i] = self.init_wolf()
        # for ind in np.nditer(
        #     self.population, flags=["refs_ok"], op_flags=["readwrite"]
        # ):
        #     ind[...] = self.init_wolf()

    def init_wolf(self):
        cells = self.problem.circle()
        fitness = self.problem.evaluate(cells)
        return Solution(cells=cells, fitness=fitness)

    def update_alpha_beta_delta(self):
        self.population = np.sort(self.population)
        self.alpha = self.population[0]
        self.beta = self.population[1]
        self.delta = self.population[2]

    def update_population(self):
        for i in range(3, self.N):
            r1 = np.random.uniform(0, 1, self.problem.size)
            r2 = np.random.uniform(0, 2, self.problem.size)
            A1 = 2 * self.a * r1 - self.a
            C1 = 2 * r2
            D_alpha = np.abs(C1 * self.alpha.cells - self.population[i].cells)
            X1 = self.alpha.cells - A1 * D_alpha

            r1 = np.random.uniform(0, 1, self.problem.size)
            r2 = np.random.uniform(0, 1, self.problem.size)
            A2 = 2 * self.a * r1 - self.a
            C2 = 2 * r2
            D_beta = np.abs(C2 * self.beta.cells - self.population[i].cells)
            X2 = self.beta.cells - A2 * D_beta

            r1 = np.random.uniform(0, 1, self.problem.size)
            r2 = np.random.uniform(0, 1, self.problem.size)
            A3 = 2 * self.a * r1 - self.a
            C3 = 2 * r2
            D_delta = np.abs(C3 * self.delta.cells - self.population[i].cells)
            X3 = self.delta.cells - A3 * D_delta

            # self.population[i].cells = np.around((X1 + X2 + X3) / 3)
            self.population[i].cells = (X1 + X2 + X3) // 3
            self.population[i].fitness = self.problem.evaluate(self.population[i].cells)

    def solve(self) -> Solution:
        self.init_population()
        self.update_alpha_beta_delta()
        best: Solution = deepcopy(self.alpha)
        it = 0
        while it < self.max_iterations:
            # self.a = 2 - it * ((2) / self.max_iterations)
            self.a = 2 - 2 * np.square(it / self.max_iterations)
            self.update_population()
            self.update_alpha_beta_delta()
            if self.alpha < best:
                best = deepcopy(self.alpha)
            self.convergence[it] = best.fitness
            if np.isclose(best.fitness, self.problem.optimal):
                return best
            it += 1
        return best

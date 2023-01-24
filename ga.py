# Genetic Algorithm
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from problem import CircleDetection
from solution import Solution


@dataclass
class GA:
    """Genetic Algorithm"""

    N: int
    generations: int
    problem: CircleDetection
    population: np.ndarray
    opponents: int

    def init_individual(self):
        cells = self.problem.circle()
        fitness = self.problem.evaluate(cells)
        return Solution(cells=cells, fitness=fitness)

    def tournament(self, num_selected: int) -> Tuple:
        selected = self.population[np.random.choice(self.N, size=(num_selected,))]
        dad = selected[0]
        mom = selected[1]
        it = 2
        while it < len(selected):
            if mom.fitness < selected[it].fitness:
                mom = selected[it]
            it += 1
        return dad, mom

    def selection(self) -> Tuple:
        dad, mom = self.tournament(self.opponents + 1)
        return dad, mom

    def split(self, chromosome: Solution) -> Tuple:
        cells = chromosome.cells
        size = cells.size
        head = cells[: size + 1 // 2]
        tail = cells[size + 1 // 2 :]
        return head, tail

    def feasible(self, head: np.ndarray, tail: np.ndarray) -> Solution:
        cells = np.append(head, tail)
        fitness = self.problem.evaluate(cells)
        child = Solution(
            cells=cells,
            fitness=fitness,
        )
        return child

    def cross(self, dad: Solution, mom: Solution) -> Tuple:
        head_dad, tail_dad = self.split(dad)
        head_mom, tail_mom = self.split(mom)
        first_child = self.feasible(head_dad, tail_mom)
        second_child = self.feasible(head_mom, tail_dad)
        return (first_child, second_child)

    def mutation(self, chromosome: Solution) -> Solution:
        mut = np.random.uniform()
        if mut < 0.05:
            chromosome.cells = self.problem.circle()
            chromosome.fitness = self.problem.evaluate(chromosome.cells)
        return chromosome

    def replace(self, population: np.ndarray) -> None:
        all = np.concatenate((self.population, population))
        self.population = np.array(sorted(all, reverse=False)[: self.N], dtype=object)

    def solve(self) -> Solution:
        self.population = np.array(
            sorted([self.init_individual() for _ in range(self.N)])
        )
        best = self.population[0]
        generation = 1
        while generation < self.generations:
            population = np.empty(shape=self.N, dtype=object)
            for i in range(0, self.N, 2):
                dad, mom = self.selection()
                first, second = self.cross(dad, mom)
                first = self.mutation(first)
                second = self.mutation(second)
                population[i] = first
                population[i + 1] = second
            self.replace(population)
            if self.population[0] < best:
                best = self.population[0]
            generation += 1
        return best

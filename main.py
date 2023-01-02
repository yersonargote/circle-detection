# -*- coding: utf-8 -*-
"""GWO-CircleDetection.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1zoJ8Cht_3oH7YjkIdrBpkgfh5EUC4u79
"""

import time
# Commented out IPython magic to ensure Python compatibility.
from dataclasses import dataclass

import cv2 as cv
import numpy as np
import typer
from matplotlib import pyplot as plt

# %matplotlib inline


@dataclass
class CircleDetection:
    name: str
    size: int
    min_radius: int
    edges: np.ndarray
    img: np.ndarray

    def select_coords(self):
        # Generate a NumPy array to store the coordinates of the three points
        coords = np.empty((3, 2))

        # Generate the three points
        while True:
            # Select three random points that have a non-zero value
            coords[:, 0] = np.random.choice(np.nonzero(self.edges)[0], 3, replace=False)
            coords[:, 1] = np.random.choice(np.nonzero(self.edges)[1], 3, replace=False)

            # Calculate the cross product of the vectors formed by the three points
            cross_product = (coords[1, 0] - coords[0, 0]) * (
                coords[2, 1] - coords[0, 1]
            ) - (coords[2, 0] - coords[0, 0]) * (coords[1, 1] - coords[0, 1])

            # If the cross product is not equal to zero, then the points are non-collinear
            if not np.isclose(cross_product, 0):
                break
        return coords

    def circle(self):
        # Init individual
        (xi, yi), (xj, yj), (xk, yk) = self.select_coords()

        # Calculate x0 and y0
        xjyj = np.square(xj) + np.square(yj)
        xiyi = np.square(xi) + np.square(yi)
        yjyi = yj - yi
        xkyk = np.square(xk) + np.square(yk)
        ykyi = yk - yi
        xjxi = xj - xi
        xkxi = xk - xi

        n1 = ((xjyj - xiyi) * (2 * ykyi)) - ((xkyk - xiyi) * (2 * yjyi))
        d = 4 * ((xjxi * ykyi) - (xkxi * yjyi))
        x0 = n1 // d

        n2 = (2 * xjxi * (xkyk - xiyi)) - (2 * xkxi * (xjyj - xiyi))
        y0 = n2 // d

        # Calculate the radius of the circle
        r = np.int64(np.sqrt((x0 - xi) ** 2 + (y0 - yi) ** 2))
        circle = np.array([x0, y0, r])
        return circle

    def evaluate(self, cells: np.ndarray) -> float:
        error = 1
        x0, y0, r = cells
        if r < self.min_radius:
            return error
        circumference = np.arange(0, (2 * np.pi) + 0.1, 0.1)
        perimeter = circumference.size
        points = 0
        for theta in circumference:
            x = int(x0 + r * np.cos(theta))
            y = int(y0 + r * np.sin(theta))
            if 0 < x < self.edges.shape[0] and 0 < y < self.edges.shape[1]:
                if self.edges[x, y] == 255:
                    points += 1
            else:
                points -= 1
        error = 1 - (points / perimeter)
        return error


@dataclass
class Solution:
    cells: np.ndarray
    fitness: float


@dataclass
class GWO:
    max_iterations: int
    N: int
    problem: CircleDetection
    population: np.ndarray
    a: float
    alpha: Solution
    beta: Solution
    delta: Solution

    def init_wolf(self):
        cells = self.problem.circle()
        fitness = self.problem.evaluate(cells)
        return Solution(cells=cells, fitness=fitness)

    def update_alpha_beta_delta(self):
        self.population = np.array(
            sorted(self.population, key=lambda x: x.fitness, reverse=False),
            dtype=object,
        )
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

            self.population[i].cells = np.around((X1 + X2 + X3) / 3)
            self.population[i].fitness = self.problem.evaluate(self.population[i].cells)

    def solve(self):
        self.population = np.array([self.init_wolf() for _ in range(self.N)])
        it = 0
        while it < self.max_iterations:
            self.a = 2 - it * ((2) / self.max_iterations)
            self.update_alpha_beta_delta()
            self.update_population()
            it += 1
        self.update_alpha_beta_delta()

        return self.alpha


def canny(filename: str):
    img = cv.imread(filename, 0)
    # edges = cv.Canny(img, 50, 50)
    edges = cv.Canny(img, 100, 200)
    return edges


def get_img(filename: str):
    img = cv.imread(filename, 0)
    img = cv.medianBlur(img, 5)
    cimg = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    return img, cimg


def show(circles: np.ndarray, edges: np.ndarray, cimg: np.ndarray):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(16, 9)

    circles = np.uint16(np.around(circles))

    for circle in circles:
        x0, y0, r = circle
        cv.circle(cimg, (x0, y0), r, (255, 0, 0), 2)
        cv.circle(cimg, (x0, y0), 2, (0, 0, 255), 3)
    ax1.imshow(cimg)
    ax1.set_title("Circle Detection")
    ax1.set_xlabel("X axis")
    ax1.set_ylabel("Y axis")

    for circle in circles:
        x0, y0, r = circle
        cv.circle(edges, (x0, y0), r, (255, 255, 255), 2)
        cv.circle(edges, (x0, y0), 2, (255, 255, 255), 3)
    ax2.imshow(edges, cmap="gray")
    ax2.set_title("Edges Circle Detection")
    ax2.set_xlabel("X axis")
    ax2.set_ylabel("Y axis")


def main(name: str = typer.Argument("2")):
    np.random.seed(42)
    filename = f"{name}.jpg"
    edges = canny(filename)
    img, cimg = get_img(filename)
    name = "Circle Detection"
    size, N, min_radius = 3, 100, 70
    problem: CircleDetection = CircleDetection(
        name=name,
        size=size,
        min_radius=min_radius,
        edges=edges,
        img=cimg,
    )

    gwo: GWO = GWO(
        max_iterations=100,
        N=N,
        problem=problem,
        population=np.empty(shape=N, dtype=object),
        a=0,
        alpha=Solution(np.zeros(size), np.Inf),
        beta=Solution(np.zeros(size), np.Inf),
        delta=Solution(np.zeros(size), np.Inf),
    )

    # Grey Wolf Optimizer
    start_time = time.perf_counter()
    best = gwo.solve()
    end_time = time.perf_counter()
    time_gwo = end_time - start_time

    # Hough Circle Transform
    start_time = time.perf_counter()
    circles = cv.HoughCircles(
        img,
        cv.HOUGH_GRADIENT,
        1,
        20,
        param1=50,
        param2=30,
        minRadius=0,
        maxRadius=0,
    )
    end_time = time.perf_counter()
    circle = circles[0, 0]
    time_hct = end_time - start_time
    other = Solution(cells=np.around(circle), fitness=problem.evaluate(circle))

    # Comparing
    print(f"GWO: {best}")
    print(f"Time GWO: {time_gwo}")
    print(f"HCT: {other}")
    print(f"Time HCT: {time_hct}")
    solutions = [solution.cells for solution in gwo.population]
    show(np.array(solutions), np.copy(edges), np.copy(cimg))
    show(np.array(circles[0, :]), np.copy(edges), np.copy(cimg))
    show(np.array([best.cells, other.cells]), np.copy(edges), np.copy(cimg))
    plt.show()


if __name__ == "__main__":
    typer.run(main)

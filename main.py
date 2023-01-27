# -*- coding: utf-8 -*-
"""GWO-CircleDetection.ipynb"""

import random
import time

import cv2 as cv
import numpy as np
import typer
from matplotlib import pyplot as plt
from rich import print

from ga import GA
from ghs import GHS
from gwo import GWO
from problem import CircleDetection
from solution import Solution


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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(19, 9))

    circles = np.uint16(circles)

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    for i, circle in enumerate(circles):
        x0, y0, r = circle
        color = colors[i % len(colors)]
        cv.circle(cimg, (x0, y0), r, color, 2)
        cv.circle(cimg, (x0, y0), 2, color, 3)

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


def show_ind(solutions: dict, img: np.ndarray):
    fig, axs = plt.subplots(2, 2, figsize=(16, 9))
    fig.tight_layout(pad=5.0)

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    it = 0
    for name, (_, solution, t) in solutions.items():
        circle = np.uint16(solution.cells)
        x0, y0, r = circle
        color = colors[it % len(colors)]
        cimg = np.copy(img)
        cv.circle(cimg, (x0, y0), r, color, 2)
        cv.circle(cimg, (x0, y0), 2, color, 3)

        axs[it // 2][it % 2].imshow(cimg)
        axs[it // 2][it % 2].set_title(
            f"{name} | Fitness: {solution.fitness} | Circle {solution.cells} | Time {t}"
        )
        axs[it // 2][it % 2].set_xlabel("X axis")
        axs[it // 2][it % 2].set_ylabel("Y axis")
        it += 1


def show_plt(solutions: dict):
    figure, axes = plt.subplots()
    axes.set_title("Circle Detection")
    colors = ["green", "blue", "red", "yellow"]
    it = 0
    for name, values in solutions.items():
        color = colors[it % len(colors)]
        axes.plot(values[0], color=color, label=name)
        it += 1
    axes.legend(loc="best")


def main(name: str = typer.Argument("2")):
    np.random.seed(42)
    random.seed(42)
    solutions = {}
    filename = f"{name}.jpg"
    edges = canny(filename)
    img, cimg = get_img(filename)
    name = "Circle Detection"
    min_radius, max_radius = (50, 150)
    optimal, size = (0.2, 3)
    N, max_iterations = (100, 100)
    # 20, 100  # 32, 89  # 14, 111 # (7, 133)  # 100, 100 # 12, 586
    problem: CircleDetection = CircleDetection(
        name=name,
        size=size,
        min_radius=min_radius,
        max_radius=max_radius,
        optimal=optimal,
        edges=edges,
        img=cimg,
    )

    gwo: GWO = GWO(
        max_iterations=max_iterations,
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
    sols, best = gwo.solve()
    end_time = time.perf_counter()
    solutions["GWO"] = (sols, best, end_time - start_time)

    # Hough Circle Transform
    start_time = time.perf_counter()
    circles = cv.HoughCircles(
        image=img,
        method=cv.HOUGH_GRADIENT,
        dp=1,
        minDist=min_radius,
        param1=50,
        param2=30,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    end_time = time.perf_counter()
    cells = np.array(np.around(circles[0, 0]))

    best = Solution(
        cells=cells,
        fitness=problem.evaluate(cells),
    )
    sols = np.array([])
    solutions["HCT"] = (sols, best, end_time - start_time)

    # Global-Harmony Search
    # HMCR, PAR = 0.9, 0.3
    HMCR, PAR = 0.7, 0.3
    BW, BWmin, BWmax = 2, 1, 5
    ghs: GHS = GHS(
        problem=problem,
        max_iterations=max_iterations,
        memory=np.empty(N, dtype=object),
        N=N,
        HMCR=HMCR,
        PAR=PAR,
        BW=BW,
        BWmin=BWmin,
        BWmax=BWmax,
    )
    start = time.perf_counter()
    sols, best = ghs.solve()
    end = time.perf_counter()
    solutions["GHS"] = (sols, best, end - start)

    # GA
    ga = GA(
        N=N,
        generations=max_iterations,
        problem=problem,
        population=np.empty(shape=N, dtype=object),
        opponents=2,
    )
    start = time.perf_counter()
    sols, best = ga.solve()
    end = time.perf_counter()
    solutions["GA"] = (sols, best, end - start)

    # Benchmarking
    for name, (sols, best, tme) in solutions.items():
        print(f"{name} - Time: {tme}")
        valid = np.isclose(problem.evaluate(best.cells), best.fitness)
        print(f"Fitness: {best.fitness} - Valid: {valid}")
        print(f"{best.cells}")
    show(
        np.array([solution[1].cells for solution in solutions.values()]),
        np.copy(edges),
        np.copy(cimg),
    )
    show_ind(solutions, np.copy(cimg))
    show_plt(solutions)
    plt.show()


if __name__ == "__main__":
    typer.run(main)

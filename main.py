# -*- coding: utf-8 -*-
"""GWO CircleDetection"""

import random
import time

import cv2 as cv
import numpy as np
import typer
from matplotlib import pyplot as plt
from rich import print

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
    colorss = ["red", "green", "blue", "yellow"]
    it = 0
    for name, (x, solution, t) in solutions.items():
        circle = np.uint16(solution.cells)
        x0, y0, r = circle
        color = colors[it % len(colors)]
        cimg = np.copy(img)
        cv.circle(cimg, (x0, y0), r, color, 2)
        cv.circle(cimg, (x0, y0), 2, color, 3)

        axs[it // 2][it % 2].imshow(cimg)
        axs[it // 2][it % 2].set_title(
            f"{name} | Fitness: {solution.fitness} | Circle {solution.cells}\nTime {t}"
        )
        axs[it // 2][it % 2].set_xlabel("X axis")
        axs[it // 2][it % 2].set_ylabel("Y axis")
        it += 1

    (x, solution, t) = solutions["GWO"]

    axs[1][0].plot(x, color=colorss[0])
    axs[1][0].set_title(
        f"""GWO - Plot
        Min {np.min(x)} - Max {np.max(x)}"""
    )
    axs[1][0].set_xlabel("Iterations")
    axs[1][0].set_ylabel("Fitness")

    axs[1][1].boxplot(x)
    axs[1][1].set_title(
        f"""GWO - Boxplot
        Mean: {np.mean(x)} - std: {np.std(x)}"""
    )
    axs[1][1].set_xlabel("")
    axs[1][1].set_ylabel("")


def main(
    img: str = typer.Argument("2"),
    n: int = typer.Argument(50),
    it: int = typer.Argument(250),
):
    np.random.seed(42)
    random.seed(42)
    solutions = {}
    filename = f"{img}.jpg"
    edges = canny(filename)
    img, cimg = get_img(filename)
    name = "Circle Detection"
    min_radius, max_radius = (50, 200)
    optimal, size = (0.0, 3)
    # 17, 153
    # 20, 100  # 32, 89  # 14, 111 # (7, 133)  # 100, 100 # 12, 586
    # N, max_iterations = (75, 48)
    N, max_iterations = n, it
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
        population=np.empty(shape=(N,), dtype=object),
        a=0,
        alpha=Solution(np.zeros(size), np.Inf),
        beta=Solution(np.zeros(size), np.Inf),
        delta=Solution(np.zeros(size), np.Inf),
        convergence=np.zeros(max_iterations),
    )

    # Grey Wolf Optimizer
    start_time = time.perf_counter()
    best = gwo.solve()
    end_time = time.perf_counter()
    sols = gwo.convergence
    solutions["GWO"] = (sols, best, end_time - start_time)
    print(f"{name} - Time: {end_time - start_time}")
    valid = np.isclose(problem.evaluate(best.cells), best.fitness)
    print(f"Fitness: {best.fitness} - Valid: {valid}")
    print(f"{best.cells}")
    print(f"Mean {np.mean(sols)} - Std: {np.mean(sols)}")
    print(f"Min {np.min(sols)} - Max: {np.max(sols)}")

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
    bests = []
    for circle in circles[0]:
        fitness = problem.evaluate(circle)
        bests.append(Solution(cells=circle, fitness=fitness))
    bests = sorted(bests)[:10]
    sols = np.array([])
    solutions["HCT"] = (sols, bests[0], end_time - start_time)
    print(f"{name} - Time: {end_time - start_time}")
    valid = np.isclose(problem.evaluate(bests[0].cells), bests[0].fitness)
    print(f"Fitness: {bests[0].fitness} - Valid: {valid}")
    print(f"{bests[0].cells}")

    show(
        np.array([solution.cells for _, solution, _ in solutions.values()]),
        np.copy(edges),
        np.copy(cimg),
    )
    show(
        np.array([solution.cells for solution in gwo.population]),
        np.copy(edges),
        np.copy(cimg),
    )
    show(
        np.array([solution.cells for solution in bests]),
        np.copy(edges),
        np.copy(cimg),
    )
    show_ind(solutions, np.copy(cimg))
    plt.show()


if __name__ == "__main__":
    typer.run(main)

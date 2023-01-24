# -*- coding: utf-8 -*-
"""GWO-CircleDetection.ipynb"""

import time

import cv2 as cv
import numpy as np
import typer
from matplotlib import pyplot as plt

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

from dataclasses import dataclass

import numpy as np


@dataclass
class CircleDetection:
    name: str
    size: int
    min_radius: int
    max_radius: int
    optimal: float
    edges: np.ndarray
    img: np.ndarray

    def select_coords(self):
        coords = np.array(np.where(self.edges == 255)).T
        while True:
            i, j, k = np.random.choice(
                list(range(len(coords))),
                size=(3,),
                replace=False,
            )
            (xi, yi), (xj, yj), (xk, yk) = coords[i], coords[j], coords[k]
            matrix = np.array(
                [
                    [1, xi, yi],
                    [1, xj, yj],
                    [1, xk, yk],
                ]
            )
            det = np.linalg.det(matrix)
            if det != 0:
                break
        return (xi, yi), (xj, yj), (xk, yk)

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
        r = np.sqrt((x0 - xi) ** 2 + (y0 - yi) ** 2)
        circle = np.array([x0, y0, r])
        return circle

    def evaluate(self, cells: np.ndarray) -> float:
        error = 1
        x0, y0, r = cells
        if r < self.min_radius or r > self.max_radius:
            return 2
        perimeter = int(2 * np.pi * r)
        circumference = np.linspace(0, 2 * np.pi, perimeter)
        points = 0
        x = (x0 + r * np.cos(circumference)).astype(int)
        y = (y0 + r * np.sin(circumference)).astype(int)
        coords = zip(y, x)
        for x, y in coords:
            if 0 < x < self.edges.shape[0] and 0 < y < self.edges.shape[1]:
                if self.edges[x][y]:
                    points += 1
        error = 1 - (points / perimeter)
        return error

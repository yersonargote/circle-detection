from dataclasses import dataclass

import numpy as np


@dataclass
class CircleDetection:
    name: str
    size: int
    min_radius: int
    max_radius: int
    optimal: int
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
        x0 = np.around(n1 / d)

        n2 = (2 * xjxi * (xkyk - xiyi)) - (2 * xkxi * (xjyj - xiyi))
        y0 = np.around(n2 / d)

        # Calculate the radius of the circle
        r = np.around(np.sqrt((x0 - xi) ** 2 + (y0 - yi) ** 2))
        circle = np.array([x0, y0, r])
        return circle

    def evaluate(self, cells: np.ndarray) -> float:
        error = 1
        x0, y0, r = cells
        if r < self.min_radius or r > self.max_radius:
            return error
        circumference = np.arange(0, (2 * np.pi) + 0.1, 0.1)
        perimeter = circumference.size
        points = 0
        for theta in circumference:
            x = int(x0 + r * np.cos(theta))
            y = int(y0 + r * np.sin(theta))
            if 0 < x < self.edges.shape[0] and 0 < y < self.edges.shape[1]:
                values = self.edges[x - 1 : x + 2, y - 1 : y + 2]
                # values = self.edges[x - 3 : x + 4, y - 3 : y + 4]
                edges = np.count_nonzero(values == 255)
                if edges > 0:
                    points += 1
                # if self.edges[x][y] == 255:
                #     points += 1
        error = 1 - (points / perimeter)
        return error

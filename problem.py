from dataclasses import dataclass

import numpy as np


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

            # If the cross product is not equal to zero,
            # then the points are non-collinear
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

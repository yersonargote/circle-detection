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
        # x0 = np.around(n1 / d)
        x0 = n1 // d

        n2 = (2 * xjxi * (xkyk - xiyi)) - (2 * xkxi * (xjyj - xiyi))
        # y0 = np.around(n2 / d)
        y0 = n2 // d

        # Calculate the radius of the circle
        r = np.around(np.sqrt((x0 - xi) ** 2 + (y0 - yi) ** 2))
        circle = np.array([x0, y0, r])
        return circle

    # def evaluate(self, cells: np.ndarray) -> float:
    #     x0, y0, r = cells
    #     if r < self.min_radius or r > self.max_radius:
    #         return 2
    #     perimeter = int(2 * np.pi * r)
    #     circumference = np.linspace(0, 2 * np.pi, perimeter)
    #     perimeter_points = np.array(
    #         [np.rint(x0 + r * np.cos(t)).astype(int) for t in circumference]
    #     )
    #     perimeter_points = np.stack(
    #         (
    #             perimeter_points,
    #             np.rint(y0 + r * np.sin(circumference)).astype(int),
    #         ),
    #         axis=1,
    #     )
    #     perimeter_points = np.unique(
    #         np.array([tuple(point) for point in perimeter_points]), axis=0
    #     )
    #     white_points = np.array(
    #         [
    #             point
    #             for point in perimeter_points
    #             if 0 < point[0] < self.edges.shape[0]
    #             and 0 < point[1] < self.edges.shape[1]
    #             and self.edges[tuple(point)]
    #         ]
    #     )
    #     fitness = 1 - (white_points.shape[0] / perimeter_points.shape[0])
    #     return fitness

    def evaluate(self, cells: np.ndarray) -> float:
        # coords = np.stack((x, y), axis=1)
        # points = [
        #     (x, y)
        #     for x, y in coords
        #     if 0 < x < self.edges.shape[0]
        #     and 0 < y < self.edges.shape[1]
        #     and self.edges[x][y]
        # ]
        # points = len(points)
        error = 1
        x0, y0, r = cells
        if r < self.min_radius or r > self.max_radius:
            return 2
        perimeter = int(2 * np.pi * r)
        circumference = np.linspace(0, 2 * np.pi, perimeter)
        points = 0
        x = np.ceil(x0 + r * np.cos(circumference)).astype(int)
        y = np.ceil(y0 + r * np.sin(circumference)).astype(int)
        for x, y in zip(x, y):
            if 0 < x < self.edges.shape[0] and 0 < y < self.edges.shape[1]:
                if self.edges[x][y]:
                    points += 1
        error = 1 - (points / perimeter)
        return error

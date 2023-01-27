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
    circumference = np.arange(0, (2 * np.pi) + 0.1, 0.1)

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
        x0 = np.abs(np.around(n1 / d))

        n2 = (2 * xjxi * (xkyk - xiyi)) - (2 * xkxi * (xjyj - xiyi))
        y0 = np.abs(np.around(n2 / d))

        # Calculate the radius of the circle
        r = np.around(np.sqrt((x0 - xi) ** 2 + (y0 - yi) ** 2))
        circle = np.array([x0, y0, r])
        return circle

    # def bresenham_circle(self, x0, y0, r) -> np.ndarray:
    #     x, y = r, 0
    #     decision = 3 - 2 * r
    #     points = []
    #     while y <= x:
    #         points.append((x + x0, y + y0))
    #         points.append((y + x0, x + y0))
    #         points.append((-x + x0, y + y0))
    #         points.append((-y + x0, x + y0))
    #         points.append((-x + x0, -y + y0))
    #         points.append((-y + x0, -x + y0))
    #         points.append((x + x0, -y + y0))
    #         points.append((y + x0, -x + y0))
    #         if decision < 0:
    #             decision = decision + 4 * y + 6
    #         else:
    #             decision = decision + 4 * (y - x) + 10
    #             x -= 1
    #         y += 1
    #     return np.array(points)

    def evaluate(self, cells: np.ndarray) -> float:
        error = 1
        x0, y0, r = cells
        if r < self.min_radius or r > self.max_radius:
            return 2
        points = 0
        perimeter = self.circumference.size
        x = np.int16(np.ceil(x0 + r * np.cos(self.circumference)))
        y = np.int16(np.ceil(y0 + r * np.sin(self.circumference)))
        for x, y in zip(x, y):
            if 0 < x < self.edges.shape[0] and 0 < y < self.edges.shape[1]:
                # values = self.edges[x - 1 : x + 2, y - 1 : y + 2]
                # values = self.edges[x - 3 : x + 4, y - 3 : y + 4]
                # edges = np.count_nonzero(values == 255)
                # if edges > 0:
                #     points += 1
                if self.edges[x][y] == 255:
                    points += 1
        if perimeter == 0:
            return 2
        error = 1 - (points / perimeter)
        return error

    # def evaluate(self, cells: np.ndarray) -> float:
    #     error = 1
    #     x0, y0, r = cells
    #     if r < self.min_radius or r > self.max_radius:
    #         return error
    #     circumference = np.arange(0, (2 * np.pi) + 0.1, 0.1)
    #     perimeter = 0
    #     points = 0
    #     x = np.int16(np.around(x0 + r * np.cos(circumference)))
    #     y = np.int16(np.around(y0 + r * np.sin(circumference)))
    #     coords = zip(x, y)
    #     for x, y in coords:
    #         if 0 < x < self.edges.shape[0] and 0 < y < self.edges.shape[1]:
    #             perimeter += 1
    #             # values = self.edges[x - 1 : x + 2, y - 1 : y + 2]
    #             # edges = np.count_nonzero(values == 255)
    #             # points += edges
    #             if self.edges[x][y] == 255:
    #                 points += 1
    #     if perimeter == 0:
    #         return 2
    #     error = 1 - (points / perimeter)
    #     return error

    # def evaluate(self, cells: np.ndarray) -> float:
    #     error = 1
    #     x0, y0, r = np.int16(cells)
    #     if r < self.min_radius or r > self.max_radius:
    #         return 2  # Infinity
    #     coords = self.bresenham_circle(x0, y0, r)
    #     coords = np.array(
    #         [
    #             self.edges[x, y]
    #             for x, y in coords
    #             if 0 < x < self.edges.shape[0] and 0 < y < self.edges.shape[1]
    #         ]
    #     )
    #     if coords.shape[0] == 0:
    #         return 2  # Infinity
    #     perimeter = coords.shape[0]
    #     points = np.count_nonzero(coords == 255)
    #     error = 1 - (points / perimeter)
    #     return error

    # def evaluate(self, cells: np.ndarray) -> float:
    #     error = 1
    #     x0, y0, r = np.int16(cells)
    #     if r < self.min_radius or r > self.max_radius:
    #         return np.Inf
    #     coords = self.bresenham_circle(x0, y0, r)
    #     coords = np.array(
    #         [
    #             (x, y)
    #             for x, y in coords
    #             if 0 < x < self.edges.shape[0] and 0 < y < self.edges.shape[1]
    #         ]
    #     )
    #     if coords.shape[0] == 0:
    #         return np.Inf
    #     perimeter = coords.shape[0]
    #     points = 0
    #     for x, y in coords:
    #         values = self.edges[x - 1 : x + 2, y - 1 : y + 2]
    #         edges = np.count_nonzero(values == 255)
    #         if edges > 0:
    #             points += 1
    #     error = 1 - (points / perimeter)
    #     return error

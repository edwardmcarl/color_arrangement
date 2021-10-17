import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from arrange_colors.color_problem import (ColorMatrix, ColorSpace, KeyPoint, solve_colors)


def run():
    np.random.seed(43)
    width = 20
    height = 40

    colors = ColorMatrix(np.random.randint(0, 256, [width * height, 3]), ColorSpace.RGB)
    
    fig, ax = plt.subplots()

    key_points = [
        KeyPoint(0, 0, np.array([255, 0, 0])),
        KeyPoint(70, 25, np.array([0, 255, 0])),
    ]

    [space_indices, color_indices] = solve_colors((width,height), colors, ColorSpace.CIELAB, key_points)


    plt.xlim(0, 10 * width)
    plt.ylim(0, 10 * height)
    ax.set_aspect("equal")

    for i in range(width):
        for j in range(height):
            ax.add_patch(
                Rectangle(
                    (i * 10, j * 10),
                    10,
                    10,
                    color=np.true_divide(colors.matrix[color_indices[j + (height * i)]], 255),
                )
            )
    plt.show()


if __name__ == "__main__":
    run()

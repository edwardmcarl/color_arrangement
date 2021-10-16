import math
from abc import ABC, abstractmethod

import numpy as np
from einops import rearrange
from skimage import color as spaces


def create_coordinate_distance_matrix(width: int, height: int, key_points):
    distances = np.empty([width, height, len(key_points)])
    for i in range(width):
        for j in range(height):
            for k in range(len(key_points)):
                distances[i][j][k] = math.sqrt(
                    ((i - key_points[k].x) ** 2) + ((j - key_points[k].y) ** 2)
                )
    new_distances = rearrange(distances, "w h k -> (w h) k")
    return new_distances


def create_color_distance_matrix(colors, key_points):
    distances = np.empty([len(colors), len(key_points)])
    for i in range(len(colors)):
        for j in range(len(key_points)):
            distances[i][j] = np.linalg.norm(colors[i] - key_points[j].lab_color)
    return distances

def create_cost_matrix(color_distances, space_distances):
    #print(space_distances)
    costs = np.empty([len(color_distances), len(space_distances)])
    for i in range(len(space_distances)):
        for j in range(len(color_distances)):
            total = 0
            for k in range(len(color_distances[0])):
                total += (color_distances[j][k] ** 2) / max(1, space_distances[i][k])
            costs[i][j] = total
    return costs


class KeyPoint:
    def __init__(self, x: int, y: int, color: np.ndarray) -> None:
        self.x = x
        self.y = y
        self.color = color
        self.lab_color = spaces.rgb2lab(color)

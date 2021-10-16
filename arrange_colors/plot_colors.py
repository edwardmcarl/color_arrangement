import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from skimage import color as spaces
from color_problem import create_color_distance_matrix, create_coordinate_distance_matrix, KeyPoint, create_cost_matrix
from scipy.optimize import linear_sum_assignment
np.random.seed(43)
width = 20
height = 20

colors = np.random.randint(0, 256, [width * height, 3])
colors_lab = spaces.rgb2lab(colors)
# print(colors_lab)
fig, ax = plt.subplots()



key_points = [KeyPoint(0, 0, np.array([255, 0, 0])), KeyPoint(70, 25, np.array([0, 255, 0]))]
space_dists = create_coordinate_distance_matrix(width, height, key_points)
color_dists = create_color_distance_matrix(colors_lab, key_points)
#print(color_dists)
#print(space_dists)
costs = create_cost_matrix(color_dists, space_dists)

[space_indices, color_indices] = linear_sum_assignment(costs)
#print(space_indices)
#print(color_indices)
plt.xlim(0, 10 * width)
plt.ylim(0, 10 * height)
ax.set_aspect("equal")

#print(colors[0])

for i in range(width):
    for j in range(height):
        #print(colors[color_indices[i + (10 * j)]])
        ax.add_patch(Rectangle((i * 10, j * 10), 10, 10, color=np.true_divide(colors[color_indices[i + (width * j)]], 255)))

plt.show()

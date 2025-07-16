import numpy as np
import matplotlib.pyplot as plt

image = np.zeros((700, 700), dtype=float)

rect_width = 500
rect_height = 200
edge_thickness = 10

center_x, center_y = 350, 350

x_start = center_x - rect_width // 2
x_end= center_x + rect_width // 2
y_start = center_y - rect_width // 2
y_end = center_y + rect_width // 2

image[y_start:y_end, x_start:x_end] = 1

inner_x_start = x_start + edge_thickness
inner_x_end = x_end - edge_thickness
inner_y_start = y_start + edge_thickness
inner_y_end = y_end - edge_thickness

image[inner_y_start:inner_y_end, inner_x_start:inner_x_end] = 0.0
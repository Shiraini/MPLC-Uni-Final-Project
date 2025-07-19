import numpy as np
import matplotlib.pyplot as plt

# Create a blank image (700x700) with zeros
image = np.zeros((700, 700), dtype=float)

# Define rectangle and border properties
rect_width = 500
rect_height = 200
edge_thickness = 10

# Center of the image
center_x, center_y = 350, 350

# Outer rectangle boundaries
x_start = center_x - rect_width // 2
x_end= center_x + rect_width // 2
y_start = center_y - rect_width // 2
y_end = center_y + rect_width // 2

# Set the outer rectangle to white (1.0)
image[y_start:y_end, x_start:x_end] = 1

# Define inner rectangle boundaries (inset by edge_thickness)
inner_x_start = x_start + edge_thickness
inner_x_end = x_end - edge_thickness
inner_y_start = y_start + edge_thickness
inner_y_end = y_end - edge_thickness

# Set the inner area back to black (0.0) to create a border effect
image[inner_y_start:inner_y_end, inner_x_start:inner_x_end] = 0.0

plt.imshow(image, cmap='gray')
plt.axis('off')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import sys

filename = "global_solution.txt"

try:
    data = np.loadtxt(filename)
except FileNotFoundError:
    print(f"Error: '{filename}' not found.")
    sys.exit(1)

plt.figure(figsize=(8, 6))
plt.imshow(data, cmap='hot', interpolation='nearest', origin='upper')

plt.colorbar(label='Potential (u)')
plt.title('2D Poisson Equation - Global Solution Heatmap')
plt.xlabel('X Grid Index')
plt.ylabel('Y Grid Index')

output_image = "poisson_heatmap.png"
plt.savefig(output_image, dpi=300, bbox_inches='tight')
print(f"Heatmap saved as '{output_image}'")
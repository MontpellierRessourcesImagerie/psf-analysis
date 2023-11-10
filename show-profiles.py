path = '/home/shaswati/Documents/PSF/60x-1.42_actual-ok/plots/radial_profiles_60x_psf_oil_actual01.json'

import json
import matplotlib.pyplot as plt
import numpy as np

# Replace 'path/to/your/file.json' with the path of your JSON file
with open(path, 'r') as f:
    data = json.load(f)

# `data` is a dictionary with integers as keys and lists of integers as values

# Preparing data for the graph
colors = plt.cm.jet(np.linspace(0, 1, len(data)))  # Create a color chart

#Creating the graph
for (key, y_values), color in zip(data.items(), colors):
    x_values = range(len(y_values)) # The abscissa are simply the rank of each ordinate 
    plt.plot(x_values, y_values, color=color)

plt.xlabel('Ordinate Index')
plt.ylabel('Value')
plt.title('Graph of ordered lists')
plt.legend()
plt.show()

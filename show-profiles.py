path = '/home/benedetti/Documents/projects/PSF/63x-confocal-ok2/plots/radial_profiles_06.json'

import json
import matplotlib.pyplot as plt
import numpy as np

# Remplacer 'path/to/your/file.json' par le chemin de votre fichier JSON
with open(path, 'r') as f:
    data = json.load(f)

# `data` est un dictionnaire avec des entiers comme clés et des listes d'entiers comme valeurs

# Préparation des données pour le graphe
colors = plt.cm.jet(np.linspace(0, 1, len(data)))  # Créer un tableau de couleurs

# Création du graphe
for (key, y_values), color in zip(data.items(), colors):
    x_values = range(len(y_values))  # Les abscisses sont simplement le rang de chaque ordonnée
    plt.plot(x_values, y_values, color=color, label=f'Clé {key}')

plt.xlabel('Index des ordonnées')
plt.ylabel('Valeur')
plt.title('Graphe des listes d\'ordonnées')
plt.legend()
plt.show()

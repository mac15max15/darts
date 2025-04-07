import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors


import display
from display import *
from constants import *

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

fname = 'sig26.9_1743875005.npy'
arr = np.load(f'heatmap_data/{fname}')

xs = np.linspace(-DOUB_OUTER-HEATMAP_PAD_MM, DOUB_OUTER+HEATMAP_PAD_MM, len(arr))
ys = np.linspace(-DOUB_OUTER-HEATMAP_PAD_MM, DOUB_OUTER+HEATMAP_PAD_MM, len(arr))

X, Y = np.meshgrid(xs, ys)

ax.plot_surface(X, Y, -1*arr, cmap='coolwarm')
plt.show()
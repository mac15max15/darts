import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors

import display
from display import *

fig, ax = display.generate_dartboard_plot()
arr = np.load('heatmap_data/sig26.9_1743875005.npy')

xs = np.linspace(-DOUB_OUTER, DOUB_OUTER, len(arr))
ys = np.linspace(-DOUB_OUTER, DOUB_OUTER, len(arr))

ax.pcolormesh(
    xs * SCALE_FACTOR, ys * SCALE_FACTOR, arr,
    shading='nearest',
    alpha=0.4,
    cmap='hot',
    #vmin=arr.min(), vmax=arr.max()
)
plt.show()

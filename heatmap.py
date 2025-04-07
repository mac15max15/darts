import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors


import display
from display import *
from constants import *

fig, ax = display.generate_dartboard_plot()
fname = 'sig26.9_1743875005'
arr = np.load(f'heatmap_data/{fname}.npy')

xs = np.linspace(-DOUB_OUTER-HEATMAP_PAD_MM, DOUB_OUTER+HEATMAP_PAD_MM, len(arr))
ys = np.linspace(-DOUB_OUTER-HEATMAP_PAD_MM, DOUB_OUTER+HEATMAP_PAD_MM, len(arr))

max_pt = np.unravel_index(np.argmax(arr), shape=arr.shape, order='F')
print(xs[max_pt[0]])
print(ys[max_pt[1]])


ax.pcolormesh(
    xs * SCALE_FACTOR, ys * SCALE_FACTOR, arr,
    shading='nearest',
    alpha=0.4,
    cmap='hot'
)
ax.scatter(xs[max_pt[0]]*SCALE_FACTOR,ys[max_pt[1]]*SCALE_FACTOR, s=100, color='darkgreen', marker='+', linewidths=1.5)
fig.savefig(f'images/{fname}.png', dpi=800)
plt.show()

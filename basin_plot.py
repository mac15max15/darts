import matplotlib.pyplot as plt
import numpy as np
from matplotlib import path
import matplotlib.patches as patches

import display
from constants import *

def parse_file(filename):
    sets = []
    accepted_flags = []

    current_set = []

    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue  # skip empty lines

            if line.endswith('*'):
                # This line ends the current set
                parts = line.split()
                accepted = line[-5] == 'T'
                accepted_flags.append(accepted)

                if current_set:
                    sets.append(np.array(current_set, dtype=float).transpose())
                    current_set = []
            else:
                # Regular data point
                try:
                    x, y = map(float, line[1:-1].split())
                    current_set.append([x, y])
                except ValueError:
                    print(f"Skipping invalid line: {line}")

    return sets, np.array(accepted_flags)



fig, ax = display.generate_dartboard_plot()
fname = 'sig10_1743922998'
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

fn = 'basin_runs/test5s10.txt'
sets, accepted_flags = parse_file(fn)


with open(fn, 'r') as file:
    l = list(map(float, ((file.readlines()[-1])[1:-1]).split()))
    ax.scatter(l[0]*SCALE_FACTOR,l[1]*SCALE_FACTOR, s=10, color='darkgreen', marker='o', linewidths=1.5)

ax.scatter(xs[max_pt[0]]*SCALE_FACTOR,ys[max_pt[1]]*SCALE_FACTOR, s=100, color='darkgreen', marker='+', linewidths=1.5)


def cm(success):
    if success:
        return 'darkgreen'
    else:
        return 'r'



# Debug print (optional)
for i, s in enumerate(sets):

    xs = SCALE_FACTOR*np.array(s[0,:])
    ys = SCALE_FACTOR*np.array(s[1,:])

    ax.plot(xs, ys, linestyle='-', marker=None, c=cm(accepted_flags[i]))
    ax.scatter(xs[-1], ys[-1], s=10, c=cm(accepted_flags[i]))
    ax.text(xs[0], ys[0], str(i+1))


plt.show()


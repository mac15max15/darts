import matplotlib.pyplot as plt
from matplotlib.patches import *
from matplotlib.path import Path
import numpy as np

from sector import *

MM_PER_INCH = 25.4
SCALE_FACTOR = 0.4 / MM_PER_INCH
PAD_WIDTH = 0.25

def main():
    fig, ax = generate_dartboard_plot()
    plt.show()


def generate_dartboard_plot():
    dim = DOUB_OUTER*2*SCALE_FACTOR
    fig, ax = plt.subplots(1, 1, figsize=(dim+PAD_WIDTH, dim+PAD_WIDTH))

    ax.set_xlim((-(SCALE_FACTOR*DOUB_OUTER+PAD_WIDTH), SCALE_FACTOR*DOUB_OUTER+PAD_WIDTH))
    ax.set_ylim((-(SCALE_FACTOR * DOUB_OUTER + PAD_WIDTH), SCALE_FACTOR * DOUB_OUTER + PAD_WIDTH))
    ax.set_axis_off()

    for radius in SCALE_FACTOR*np.array([BULL_INNER , BULL_OUTER, TRIP_INNER, TRIP_OUTER, DOUB_INNER, DOUB_OUTER]):
        ax.add_patch(Circle((0, 0), radius=radius, fill=False))

    for i in range(20):
        ang = -WIDTH/2 + i*WIDTH
        path = Path(np.array([
            (SCALE_FACTOR*BULL_OUTER*np.cos(ang), SCALE_FACTOR*BULL_OUTER*np.sin(ang)),
            (SCALE_FACTOR*DOUB_OUTER*np.cos(ang), SCALE_FACTOR*DOUB_OUTER*np.sin(ang))
        ]))

        ax.add_patch(PathPatch(path))

    return fig, ax


if __name__ == '__main__':
    main()





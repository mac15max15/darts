import matplotlib.pyplot as plt
from matplotlib.patches import *
from matplotlib.path import Path

from src.math.sector import *
"""
This file contains the code for generating a blank dartboard plot
"""
def main():
    fig, ax = generate_dartboard_plot()
    plt.show()


def generate_dartboard_plot() -> tuple[plt.Figure, plt.Axes]:
    """
    Generate a blank dartboard matplotlib figure+axis.
    :return: the figure and the axis
    """

    dim = (DOUB_OUTER+HEATMAP_PAD_MM)*SCALE_FACTOR*2
    fig, ax = plt.subplots(1, 1, figsize=(dim, dim))

    ax.set_axis_off()

    # rings
    for radius in SCALE_FACTOR*np.array([BULL_INNER , BULL_OUTER, TRIP_INNER, TRIP_OUTER, DOUB_INNER, DOUB_OUTER]):
        ax.add_patch(Circle((0, 0), radius=radius, fill=False))

    # radial lines
    for i in range(20):
        ang = -WIDTH/2 + i*WIDTH
        path = Path(np.array([
            (SCALE_FACTOR*BULL_OUTER*np.cos(ang), SCALE_FACTOR*BULL_OUTER*np.sin(ang)),
            (SCALE_FACTOR*DOUB_OUTER*np.cos(ang), SCALE_FACTOR*DOUB_OUTER*np.sin(ang))
        ]))

        ax.add_patch(PathPatch(path))

    # numbers
    for i, val in enumerate(SECTOR_VALUES):
        ang = i*WIDTH
        ax.text(
            1.05*np.cos(ang)*DOUB_OUTER*SCALE_FACTOR,
            1.05*np.sin(ang)*DOUB_OUTER*SCALE_FACTOR,
            str(val),
            horizontalalignment='center',
            verticalalignment='center',
            rotation=(ang*180/np.pi)-90
        )



    return fig, ax


if __name__ == '__main__':
    main()





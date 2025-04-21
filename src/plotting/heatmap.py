from src.math.distribution import *
from src.math.constants import *

import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import time


def plot_heatmap(arr, padding=HEATMAP_PAD_MM, save_fig=False, fname=None, show=True):
    """
    Plot a heatmap from a numpy array
    :param arr: array of expected score data
    :param padding: dont touch this its poisonous
    :param save_fig: do you wanna save the figure
    :param fname: filename for if youre saving the figure
    :param show: do you wanna show the figure

    """
    fig, ax = generate_dartboard_plot()

    xs = np.linspace(-DOUB_OUTER-padding, DOUB_OUTER+padding, len(arr))
    ys = np.linspace(-DOUB_OUTER-padding, DOUB_OUTER+padding, len(arr))

    # extract mu*
    max_pt = np.unravel_index(np.argmax(arr), shape=arr.shape, order='F')
    print(f'Actual mu*: ({xs[max_pt[0]]:.2f}, {ys[max_pt[1]]:.2f}), ')
    heatmap = ax.pcolormesh(
        xs * SCALE_FACTOR, ys * SCALE_FACTOR, arr,
        shading='nearest',
        alpha=0.4,
        cmap='gist_heat'
    )

    cax = inset_axes(
        ax,
        width="3%",
        height="100%",
        loc='lower left',
        borderpad=0,
        bbox_to_anchor=(1.01, 0., 1, 1),
        bbox_transform=ax.transAxes
    )

    fig.colorbar(heatmap, cax=cax, label='Expected Score')

    ax.scatter(xs[max_pt[0]]*SCALE_FACTOR, ys[max_pt[1]]*SCALE_FACTOR, s=100, color='darkgreen', marker='+', linewidths=1.5)
    if save_fig:
        if not fname:
            fname = int(time.time())
        fig.savefig(f'images/{fname}', dpi=800)

    if show:
        plt.show()

if __name__ == '__main__':
    arr = np.load('/Users/maxcaragozian/Desktop/MATH 305/Darts/heatmap_data/high_res_26.9.npy')
    plot_heatmap(arr, fname='gistheat.png', save_fig=False)
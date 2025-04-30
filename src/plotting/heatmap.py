from src.math.distribution import *
from src.math.constants import *
from src.plotting.display import *

import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import time


def plot_heatmap(arr, padding=HEATMAP_PAD_MM, save_fig=False, filename=None, show=True, sigma=0, title=None):
    """
    Plot a heatmap from a numpy array
    :param arr: array of expected score data
    :param padding: dont touch this its poisonous
    :param save_fig: flag for whether to save the figure. if this is ture you must also provide a filename
    :param filename: filename
    :param show: flag for whether to show the figure

    """
    fig, ax = generate_dartboard_plot()

    xs = np.linspace(-DOUB_OUTER-padding, DOUB_OUTER+padding, len(arr))
    ys = np.linspace(-DOUB_OUTER-padding, DOUB_OUTER+padding, len(arr))

    # extracting and plotting mu*
    max_pt = np.unravel_index(np.argmax(arr), shape=arr.shape, order='F')
    ax.scatter(xs[max_pt[0]]*SCALE_FACTOR, ys[max_pt[1]]*SCALE_FACTOR, s=100, color='darkgreen', marker='+', linewidths=1.5)
    print(f'Estimated mu*: ({xs[max_pt[0]]:.2f}, {ys[max_pt[1]]:.2f}), ')

    heatmap = ax.pcolormesh(
        xs * SCALE_FACTOR, ys * SCALE_FACTOR, arr,
        shading='nearest',
        alpha=0.4,
        cmap='gist_heat'
    )

    # colorbar stuff
    cax = inset_axes(
        ax,
        width="2%",
        height="100%",
        loc='lower left',
        borderpad=0,
        bbox_to_anchor=(1.005, 0., 1, 1),
        bbox_transform=ax.transAxes
    )
    plt.rcParams['text.usetex'] = True
    ticks = np.arange(np.min(arr), np.max(arr), (np.max(arr) - np.min(arr))//5)
    fig.colorbar(heatmap, cax=cax, label=rf'Expected Score, $F(\mu \vert \sigma = {sigma})$', ticks=ticks)

    if title:
        ax.set_title(title)

    if save_fig and filename:
        fig.savefig(f'../../images/{filename}', dpi=800)

    if show:
        plt.show()

if __name__ == '__main__':
    arr = np.load('/Users/maxcaragozian/Desktop/MATH 305/Darts/heatmap_data/high_res_26.9.npy')
    plot_heatmap(arr, filename='gist_hear.png', save_fig=True, sigma=26.9)
from src.math.distribution import *
import numpy as np

def plot_heatmap(arr, padding=HEATMAP_PAD_MM, save_fig=False, fname=None):
    fig, ax = generate_dartboard_plot()

    xs = np.linspace(-DOUB_OUTER-padding, DOUB_OUTER+padding, len(arr))
    ys = np.linspace(-DOUB_OUTER-padding, DOUB_OUTER+padding, len(arr))

    max_pt = np.unravel_index(np.argmax(arr), shape=arr.shape, order='F')
    print(xs[max_pt[0]])
    print(ys[max_pt[1]])

    ax.pcolormesh(
        xs * SCALE_FACTOR, ys * SCALE_FACTOR, arr,
        shading='nearest',
        alpha=0.4,
        cmap='gist_heat'
    )
    ax.scatter(xs[max_pt[0]]*SCALE_FACTOR,ys[max_pt[1]]*SCALE_FACTOR, s=100, color='darkgreen', marker='+', linewidths=1.5)

    if save_fig:
        if not fname:
            fname = int(time.time())
        fig.savefig(f'images/{fname}.png', dpi=DPI)
    plt.show()

if __name__ == '__main__':
    arr = np.load('/Users/maxcaragozian/Desktop/MATH 305/Darts/heatmap_data/high_res_26.9.npy')
    plot_heatmap(arr, fname='cividistest.png')
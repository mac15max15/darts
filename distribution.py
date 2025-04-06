import numpy as np
import scipy as spi
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

from sector import *
from display import *
from constants import *


sectors = get_sectors()


def main():
    max_pt = find_best_multinormal_center_bf(10, 150, generate_heatmap=True, disp_pct=True)
    print(max_pt)
    plt.show()


def find_best_multinormal_center_hopping(stdev):
    return spi.optimize.basinhopping(
        lambda x: -calculate_dist_ev(
            generate_symmetric_distribution(x[0], x[1], stdev)),
        x0=np.array([0, 0]),
        stepsize=DOUB_OUTER
    ).x

def find_best_multinormal_center_bf(stdev, pts=100, generate_heatmap=False, disp_pct=False):

    xs = np.linspace(-DOUB_OUTER-HEATMAP_PAD_MM, DOUB_OUTER+HEATMAP_PAD_MM, pts)
    ys = np.linspace(-DOUB_OUTER-HEATMAP_PAD_MM, DOUB_OUTER+HEATMAP_PAD_MM, pts)


    results = np.ndarray((pts, pts))
    max_ev = 0
    max_ev_loc = (0, 0)

    for i, x in enumerate(xs):
        if disp_pct:
            print(f'{100 * i / pts:.2f}%')
        for j, y in enumerate(ys):
            dist = generate_symmetric_distribution(x, y, stdev)
            results[i][j] = calculate_dist_ev(dist)

            if results[i][j] > max_ev:
                max_ev = results[i][j]
                max_ev_loc = (x, y)

    if generate_heatmap:
        fig, ax = generate_dartboard_plot()
        ax.pcolormesh(xs * SCALE_FACTOR, ys * SCALE_FACTOR, results.transpose(), shading='nearest', alpha=0.3)
        ax.scatter(max_ev_loc[0] * SCALE_FACTOR, max_ev_loc[1] * SCALE_FACTOR, color='r')
        fig.savefig(f'images/sig{stdev}_{int(time.time())}.png', dpi=800)
        np.save(f'heatmap_data/sig{stdev}_{int(time.time())}.npy', results.transpose())

    return max_ev_loc, max_ev


def calculate_dist_ev(dist, x=None, y=None, stdev=None):
    """
    Calculate the expected score from a dart thrown with a given random distribution
    :param dist: the distribution of where the dart will land
    :return: the expected value of the dart
    """

    ev = 0
    for sec in sectors:
        ev += calculate_sector_ev(sec, dist)

    return ev


def calculate_sector_ev(sec: Sector, dist):
    """
    Given a random distribution and a sector of the dartboard. Integrate over
    the sector to find the expected score it contributes.
    :param sec: The sector to integrate over
    :param dist: The distribution to integrate over
    :return: The expected value contributed by sec
    """

    if dist.pdf(sec.get_midpoint()) < 1e-20:
        return 0

    return spi.integrate.nquad(
        lambda r, theta: dist.pdf((r * np.cos(theta), r * np.sin(theta))) * sec.val * r,
        ranges=[(sec.r_min, sec.r_max), (sec.theta_min, sec.theta_max)]
    )[0]


def generate_symmetric_distribution(x, y, stdev):
    return spi.stats.multivariate_normal((x, y), get_covariance_mat(stdev))


def get_covariance_mat(stdev):
    return np.eye(2) * (stdev ** 2)


class UniformDist:
    def pdf(self, x):
        return 1 / (np.pi * 170 * 170)


if __name__ == "__main__":
    main()

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
    sigmas = np.linspace(49, 100, 52)
    t = int(time.time())

    for sigma in sigmas:
        res, val, grid, jout = find_best_multinormal_center_numpy_bf(sigma, 25)
        with open(f'results/res1743995606.txt', 'a') as res_file:
            res_file.write(f'{sigma},{res[0]},{res[1]},{-val}\n')
        print(sigma)



def find_best_multinormal_center_numpy_bf(stdev, n):
    """
    Find the point (x, y) that maximizes the expected score for a dart thrown with
    a symmetrical normal distribution centered on (x, y) and with standard deviation
    stdev.

    This method uses SciPy's brute force global optimizer which creates an n by n
    grid of points and evaluates the function (ie the expected value of the distribution)
    at each point. After the brute force, it then uses a gradient-based local optimizer (spi.optimize.fmin)
    to refine the point.
    """
    return spi.optimize.brute(
        lambda x: -calculate_dist_ev(
            generate_symmetric_distribution(x[0], x[1], stdev)),
        ranges=((-DOUB_OUTER, DOUB_OUTER), (-DOUB_OUTER, DOUB_OUTER)),
        Ns=n,
        finish=spi.optimize.fmin,
        full_output=True
    )

def find_best_multinormal_center_diffev(stdev):
    return spi.optimize.shgo(
        lambda x: -calculate_dist_ev(
            generate_symmetric_distribution(x[0], x[1], stdev)),
        bounds=((-DOUB_OUTER, DOUB_OUTER), (-DOUB_OUTER, DOUB_OUTER))
    ).x

def find_best_multinormal_center_hopping(stdev):
    return spi.optimize.basinhopping(
        lambda x: -calculate_dist_ev(
            generate_symmetric_distribution(x[0], x[1], stdev)),
        x0=np.array([0, 0]),
        niter=20,
        stepsize=150,
        T=1,
        callback= lambda x, f, accept: basin_iter_callback,
        minimizer_kwargs={'callback': minimizer_callback}
    ).x

def basin_iter_callback(x, f, accept):
    with open('basin_runs/test0.txt', 'a') as out:
        out.write(f'Point: {x}, Function Value:{f}, Accepted?: {accept}*\n')

def minimizer_callback(x):
    with open('basin_runs/test0.txt', 'a') as out:
        out.write(f'{x}\n')


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


def calculate_dist_ev(dist):
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

    if sec.get_sector_approx_max(dist.pdf, 3) < SECTOR_PDF_IGNORE_THRESHOLD:
        return 0

    return spi.integrate.nquad(
        lambda r, theta: dist.pdf((r * np.cos(theta), r * np.sin(theta))) * sec.val * r,
        ranges=[(sec.r_min, sec.r_max), (sec.theta_min, sec.theta_max)],
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

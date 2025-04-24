import pandas as pd
import scipy as spi
import time

from src.plotting.display import *
from src.plotting.heatmap import *
from src.math.constants import *


sectors = get_sectors()
monte_carlo_n = 100
of = 'output.txt'


def main():
    arr = compute_grid_convolve(10, 300)

    plot_heatmap(arr, fname='s10.png', save_fig=True, sigma=10)

    arr = compute_grid_convolve(30, 300)

    plot_heatmap(arr, fname='s30.png', save_fig=True, sigma=30)


def compute_grid_brute(stdev, ev_method, n=300, mn=100):
    """
    Compute the expected score of a symmetrical distribution over a grid
    of points

    :param stdev: standard deviation
    :param ev_method: method to calculate expected score of the distribution (integration or monte-carlo)
    :param n: side length of the grid
    :return: grid of expected scores
    """
    global monte_carlo_n
    monte_carlo_n=mn

    coords = np.linspace(-DOUB_OUTER-HEATMAP_PAD_MM, DOUB_OUTER+HEATMAP_PAD_MM, n)
    func = np.vectorize(lambda x, y: ev_method(generate_symmetric_distribution(x, y, stdev)))
    x, y = np.meshgrid(coords, coords)

    return func(x, y)

def compute_grid_convolve(stdev, n=100, save_data=False, filename=None):
    dist = generate_symmetric_distribution(0, 0, stdev)
    coords = np.linspace(-DOUB_OUTER-HEATMAP_PAD_MM, DOUB_OUTER+HEATMAP_PAD_MM, n)

    xv, yv = np.meshgrid(coords, coords)

    score_grid = np.vectorize(get_score)(xv, yv)
    pdf_grid = dist.pdf(np.stack([xv, yv], axis=-1))

    dx = coords[1] - coords[0]
    mass = np.sum(pdf_grid) * dx * dx
    pdf_grid /= mass

    arr = spi.signal.fftconvolve(score_grid, pdf_grid, mode='same')
    arr *= dx * dx

    if save_data:
        if filename:
            np.save(filename, arr)
        else:
            np.save(f'sig{stdev}_{n}pts_{int(time.time())}.npy', arr)

    return arr


def find_best_multinormal_center_hopping(
        stdev,
        t=1,
        niter_sucess=5,
        stepsize=150,
        filename=None):

    if filename:
        global of
        of = filename

    return spi.optimize.basinhopping(
        lambda x: -calculate_dist_ev_integration(
            generate_symmetric_distribution(x[0], x[1], stdev)
        ),
        x0=np.array([0, 0]),
        stepsize=stepsize,
        T=t,
        niter_success=niter_sucess,
        callback=basin_iter_callback,
        minimizer_kwargs={
            'callback': minimizer_callback,
            'method': 'CG'
        }
    ).x


def basin_iter_callback(x, f, accept):
    with open(of, 'a') as out:
        out.write(f'Point: {x}, Function Value:{f}, Accepted?: {accept}*\n')

def minimizer_callback(x):
    with open(of, 'a') as out:
        out.write(f'{x}\n')

def calculate_dist_ev_monte_carlo(dist):
    """
    Calculate the expected score from a dart thrown with a given random distribution
    by random sampling
    :param dist: distribution
    :param n: number of random samples
    :return: expected score
    """
    func = np.vectorize(get_score)
    rvs = dist.rvs(monte_carlo_n).transpose()
    return np.mean(func(rvs[0], rvs[1]))

def calculate_dist_ev_integration(dist):
    """
    Calculate the expected score from a dart thrown with a given random distribution
    by direct integration
    :param dist: the distribution of where the dart will land
    :return: the expected score of the dart
    """

    func = np.vectorize(lambda sec: calculate_sector_ev_integration(sec, dist))
    return np.sum(func(np.array(get_sectors())))


def calculate_sector_ev_integration(sec: Sector, dist):
    """
    Given a random distribution and a sector of the dartboard. Integrate over
    the sector to find the expected score it contributes.
    :param sec: The sector to integrate over
    :param dist: The distribution to integrate over
    :return: The expected value contributed by sec
    """

    # if sec.get_sector_approx_max(dist.pdf, 3) < SECTOR_PDF_IGNORE_THRESHOLD:
    #     return 0

    return spi.integrate.nquad(
        lambda r, theta: dist.pdf((r * np.cos(theta), r * np.sin(theta))) * sec.val * r,
        ranges=[(sec.r_min, sec.r_max), (sec.theta_min, sec.theta_max)],
    )[0]


def generate_symmetric_distribution(x, y, stdev):
    return spi.stats.multivariate_normal((x, y), get_covariance_mat(stdev))


def get_covariance_mat(stdev):
    return np.eye(2) * (stdev ** 2)


if __name__ == "__main__":
    main()

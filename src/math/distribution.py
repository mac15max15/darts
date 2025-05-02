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
    pass


def compute_grid_brute(stdev, ev_method, n=300, mn=100):
    """
    Compute the expected score of a darth thrown with a symmetrical distribution over a grid
    of points using either integration or monte-carlo.

    :param stdev: standard deviation
    :param ev_method: method to calculate expected score of the distribution (integration or monte-carlo)
    :param n: side length of the grid
    :param mn: Number of samples for each function evaluation (for use with monte carlo)
    :return: grid of expected scores
    """
    global monte_carlo_n
    monte_carlo_n = mn

    coords = np.linspace(-DOUB_OUTER-HEATMAP_PAD_MM, DOUB_OUTER+HEATMAP_PAD_MM, n)
    func = np.vectorize(lambda x, y: ev_method(generate_symmetric_distribution(x, y, stdev)))
    x, y = np.meshgrid(coords, coords)

    return func(x, y)

def compute_grid_convolve(stdev, n=100):
    """
    Compute the expected score of a darth thrown with a symmetrical distribution over a grid
    of points using a convolution
    :param stdev: standard deviation
    :param n: side length of the grid.
    :return: grid of expected scores
    """
    dist = generate_symmetric_distribution(0, 0, stdev)
    coords = np.linspace(-DOUB_OUTER-HEATMAP_PAD_MM, DOUB_OUTER+HEATMAP_PAD_MM, n)

    xv, yv = np.meshgrid(coords, coords)

    score_grid = np.vectorize(get_score)(xv, yv)
    pdf_grid = dist.pdf(np.stack([xv, yv], axis=-1))

    dx = coords[1] - coords[0]
    mass = np.sum(pdf_grid) * dx * dx
    pdf_grid /= mass

    arr = spi.signal.fftconvolve(score_grid, pdf_grid, mode='same')
    arr *= dx * dx  # scale the results by the area of each cell

    return arr


def find_best_multinormal_center_hopping(
        stdev,
        t=None,
        niter_sucess=5,
        stepsize=150,
        filename=None):
    """
    Compute the optimal place to aim for a symettrical distribution using basin hopping.
    Write the progress to a file. See the scipy.optimize.basinhopping documentation for
    what t, niter_sucess, and stepsize to.

    :param stdev: standard deviation
    :param filename: file name for basin hopping record
    :return: the optimal point
    """
    if not t:
        t = 100/stdev

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
    Calculate the expected score from a single dart thrown with a given random distribution
    by random sampling
    :param dist: distribution
    :param n: number of random samples
    :return: expected score
    """
    get_score_vectorized = np.vectorize(get_score)
    samples = dist.rvs(monte_carlo_n).transpose()
    return np.mean(get_score_vectorized(samples[0], samples[1]))

def calculate_dist_ev_integration(dist):
    """
    Calculate the expected score from a dart thrown with a given random distribution
    by direct integration
    :param dist: the distribution of where the dart will land
    :return: the expected score of the dart
    """

    integrate_sector_vectorized = np.vectorize(lambda sec: calculate_sector_ev_integration(sec, dist))
    return np.sum(integrate_sector_vectorized(np.array(get_sectors())))


def calculate_sector_ev_integration(sec: Sector, dist):
    """
    Given a random distribution and a sector of the dartboard. Integrate over
    the sector to find the expected score it contributes.
    :param sec: The sector to integrate over
    :param dist: The distribution to integrate over
    :return: The expected value contributed by sec
    """

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

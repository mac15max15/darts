import pandas as pd
import scipy as spi
import time

from src.plotting.display import *
from src.plotting.heatmap import *
from src.math.constants import *


sectors = get_sectors()
of = 'output.txt'


def main():
    # sigmas = np.arange(16, 17, .05)
    # df = pd.DataFrame(
    #     index=sigmas,
    #     columns=['x', 'y', 'ev']
    # )
    #
    # for sigma in sigmas:
    #     center, ev = find_best_multinormal_center_convolve(sigma, 500)
    #     df.loc[sigma] = {'x': center[0], 'y': center[1], 'ev': ev}
    #     print(sigma)
    #
    # df.to_csv(f'close_run.csv')

    x0, fval, grid, jout = compute_grid_brute(19, calculate_dist_ev_monte_carlo, 30)
    plot_heatmap(-jout.transpose())


def compute_grid_brute(stdev, ev_method, n=300):
    """
    Compute the expected score of a symmetrical distribution over a grid
    of

    This method uses SciPy's brute force global optimizer which creates an n by n
    grid of points and evaluates the function (ie the expected value of the distribution)
    at each point. After the brute force, it then uses a gradient-based local optimizer (spi.optimize.fmin)
    to refine the point.
    :param stdev: standard deviation
    :param ev_method: method to calculate expected score of the distribution
    :param n: side length of the grid
    :return: the optimizer output (see spi documentation for details)
    """
    dim = DOUB_OUTER+HEATMAP_PAD_MM
    return spi.optimize.brute(
        lambda x: -ev_method(
            generate_symmetric_distribution(x[0], x[1], stdev)),
        ranges=((-dim, dim), (-dim, dim)),
        Ns=n,
        finish=spi.optimize.fmin,
        full_output=True
    )

def compute_grid_convolve(stdev, n=100, save_data=False, filename=None):
    dist = generate_symmetric_distribution(0, 0, stdev)
    grid_extent = DOUB_OUTER + HEATMAP_PAD_MM

    x = np.linspace(-grid_extent, grid_extent, n)
    y = np.linspace(-grid_extent, grid_extent, n)
    xv, yv = np.meshgrid(x, y)

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    score_grid = np.vectorize(get_score)(xv, yv)
    coord_grid = np.stack((xv.ravel(), yv.ravel()), axis=-1)
    pdf_grid = np.array([dist.pdf(pt) for pt in coord_grid]).reshape(xv.shape)
    mass = np.sum(pdf_grid) * dx * dy
    pdf_grid /= mass

    arr = spi.signal.fftconvolve(score_grid, pdf_grid, mode='same')
    arr *= dx * dy

    if save_data:
        if filename:
            np.save(filename, arr)
        else:
            np.save(f'heatmap_data/convolution/sig{stdev}_{n}pts_{int(time.time())}.npy', arr)

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
        lambda x: -calculate_dist_ev(
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


def find_best_multinormal_center_convolve(stdev, n=300):
    arr = compute_grid_convolve(stdev, n)
    coords = np.linspace(-DOUB_OUTER - HEATMAP_PAD_MM, DOUB_OUTER + HEATMAP_PAD_MM, n)
    idx_max, max = np.unravel_index(np.argmax(arr), shape=arr.shape, order='F'), np.max(arr)
    return (coords[idx_max[0]], coords[idx_max[1]]), max


def basin_iter_callback(x, f, accept):
    with open(of, 'a') as out:
        out.write(f'Point: {x}, Function Value:{f}, Accepted?: {accept}*\n')

def minimizer_callback(x):
    with open(of, 'a') as out:
        out.write(f'{x}\n')



def calculate_dist_ev_monte_carlo(dist, n=100):
    return np.mean([get_score(x0[0], x0[1]) for x0 in dist.rvs(n)])

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

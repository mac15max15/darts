import pandas as pd
import scipy as spi
import time

from src.plotting.display import *
from src.plotting.heatmap import *
from src.math.constants import *


sectors = get_sectors()
of = 'fasdfas.txt'


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


    arr = generate_heatmap_data_convolve(10, 300)
    np.save('../../heatmap_data/s10highres.npy', arr)



def find_best_multinormal_center_numpy_bf(stdev, n=300):
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
        full_output=False
    )


def find_best_multinormal_center_hopping(stdev, t=1, niter_sucess=5, stepsize=150):
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
    arr = generate_heatmap_data_convolve(stdev, n)
    coords = np.linspace(-DOUB_OUTER - HEATMAP_PAD_MM, DOUB_OUTER + HEATMAP_PAD_MM, n)
    idx_max, max = np.unravel_index(np.argmax(arr), shape=arr.shape, order='F'), np.max(arr)
    return (coords[idx_max[0]], coords[idx_max[1]]), max


def basin_iter_callback(x, f, accept):
    with open(of, 'a') as out:
        out.write(f'Point: {x}, Function Value:{f}, Accepted?: {accept}*\n')

def minimizer_callback(x):
    with open(of, 'a') as out:
        out.write(f'{x}\n')


def generate_heatmap_data_integration(stdev, n=100, disp_pct=False):

    xs = np.linspace(-DOUB_OUTER - HEATMAP_PAD_MM, DOUB_OUTER + HEATMAP_PAD_MM, n)
    ys = np.linspace(-DOUB_OUTER - HEATMAP_PAD_MM, DOUB_OUTER + HEATMAP_PAD_MM, n)

    results = np.ndarray((n, n))
    max_ev = 0
    max_ev_loc = (0, 0)

    for i, x in enumerate(xs):
        if disp_pct:
            print(f'{100 * i / n:.2f}%')
        for j, y in enumerate(ys):
            dist = generate_symmetric_distribution(x, y, stdev)
            results[i][j] = calculate_dist_ev(dist)

            if results[i][j] > max_ev:
                max_ev = results[i][j]
                max_ev_loc = (x, y)

    np.save(f'heatmap_data/integration/sig{stdev}_{n}pts_{int(time.time())}.npy', results.transpose())

    return max_ev_loc, max_ev

def generate_heatmap_data_convolve(stdev, n=100, save_data=False):
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
        np.save(f'heatmap_data/convolution/sig{stdev}_{n}pts_{int(time.time())}.npy', arr)

    return arr

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

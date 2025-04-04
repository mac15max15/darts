import numpy as np
import scipy as spi
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from sector import *
from display import *

sectors = get_sectors()
def main():
    print(generate_stdev_heatmap(10, pts=3))


def generate_stdev_heatmap(stdev, pts=100):
    xs = np.linspace(-DOUB_OUTER, DOUB_OUTER, pts)
    ys = np.linspace(-DOUB_OUTER, DOUB_OUTER, pts)

    cell_width = 2*DOUB_OUTER/pts
    fig, ax = generate_dartboard_plot()

    results = np.ndarray((pts, pts))

    for i, x in enumerate(xs):
        print(f'{100*i/pts}%')
        for j, y in enumerate(ys):
            dist = spi.stats.multivariate_normal((x, y), get_covariance_mat(stdev))
            if dist.pdf((x, y)) < 1e-3:
                results[i][j]=0
            else:
                results[i][j] = calculate_dist_ev(dist)

            cell_patch = patches.Rectangle(
                ((x-cell_width/2)*SCALE_FACTOR, (y-cell_width/2)*SCALE_FACTOR),
                cell_width*SCALE_FACTOR, cell_width*SCALE_FACTOR,
                alpha = max(0, (np.log10(results[i][j])+7)/10),
                color='yellow'
            )
            ax.add_patch(cell_patch)

    plt.show()
    return results




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
    return spi.integrate.nquad(
        lambda r, theta: dist.pdf((r*np.cos(theta), r*np.sin(theta)))*sec.val*r,
        ranges=[(sec.r_min, sec.r_max), (sec.theta_min, sec.theta_max)]
    )[0]


def get_covariance_mat(stdev):
    return np.eye(2)*(stdev**2)


class UniformDist:
    def pdf(self, x):
        return 1/(np.pi*170*170)


if __name__ == "__main__":
    main()





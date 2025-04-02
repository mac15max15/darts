import numpy as np
import scipy as spi


from sector import *


def main():
    print(expected_value(0, 103, 1))

def get_covariance_mat(stdev):
    return np.eye(2)*(stdev**2)


def expected_value(mu_x, mu_y, stdev):
    cov = get_covariance_mat(stdev)
    dist = spi.stats.multivariate_normal((mu_x, mu_y), cov)
    res = spi.integrate.nquad(
        lambda r, theta: dist.pdf((r*np.cos(theta), r*np.sin(theta)))*20*r,
        ranges=[(99, 107), (1.4137, 1.728)]
    )
    return res

if __name__ == "__main__":
    main()





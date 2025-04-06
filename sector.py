from dataclasses import dataclass
import math
import numpy as np
from itertools import product

from constants import *



def ptoc(polar_coords):
    r = polar_coords[:, 0]
    theta = polar_coords[:, 1]
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack((x, y))

@dataclass
class Sector:
    theta_min: float
    theta_max: float
    r_min: float
    r_max: float
    val: int

    def get_midpoint(self):
        r_mean, theta_mean = np.mean([self.r_min, self.r_max]), np.mean([self.theta_min, self.theta_max])
        return r_mean*np.cos(theta_mean), r_mean*np.sin(theta_mean)

    def get_sector_approx_max(self, pdf, n):

        pts = cartesian_product(
                np.linspace(self.r_min, self.r_max, n),
                np.linspace(self.theta_min, self.theta_max, n)
        )
        return np.max(pdf(ptoc(pts)))



# stackoverflow.com/questions/11144513/cartesian-product-of-x-and-y-array-points-into-single-array-of-2d-points
def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

def get_sectors():
    # inner and outer bulls
    sectors = [Sector(0, 2 * np.pi, 0, BULL_INNER, 50), Sector(0, 2 * np.pi, BULL_INNER, BULL_OUTER, 25)]

    for i, val in enumerate(SECTOR_VALUES):
        theta_min = i * WIDTH - WIDTH / 2
        theta_max = (i + 1) * WIDTH - WIDTH / 2

        sectors.append(Sector(theta_min, theta_max, BULL_OUTER, TRIP_INNER, val))
        sectors.append(Sector(theta_min, theta_max, TRIP_INNER, TRIP_OUTER, val * 3))
        sectors.append(Sector(theta_min, theta_max, TRIP_OUTER, DOUB_INNER, val))
        sectors.append(Sector(theta_min, theta_max, DOUB_INNER, DOUB_OUTER, val * 2))

    return sectors

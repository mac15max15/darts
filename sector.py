from dataclasses import dataclass
import math
import numpy as np

WIDTH = 2 * np.pi / 20
BULL_INNER = 12.7
BULL_OUTER = 32
TRIP_INNER = 99
TRIP_OUTER = 107
DOUB_INNER = 162
DOUB_OUTER = 170

# The numbers on a standard dartboard
SECTOR_VALUES = [6, 13, 4, 18, 1, 20, 5, 12, 9, 14, 11, 8, 16, 7, 19, 3, 17, 2, 15, 10]


@dataclass
class Sector:
    theta_min: float
    theta_max: float
    r_min: float
    r_max: float
    val: int

def get_sectors():
    sectors = [Sector(0, 2 * np.pi, 0, BULL_INNER, 50), Sector(0, 2 * np.pi, BULL_INNER, BULL_OUTER, 25)]
    # Add the bullseye

    # Add the segments
    for i, val in enumerate(SECTOR_VALUES):
        theta_min = i * WIDTH - WIDTH / 2
        theta_max = (i + 1) * WIDTH - WIDTH / 2

        sectors.append(Sector(theta_min, theta_max, BULL_OUTER, TRIP_INNER, val))
        sectors.append(Sector(theta_min, theta_max, TRIP_INNER, TRIP_OUTER, val * 3))
        sectors.append(Sector(theta_min, theta_max, TRIP_OUTER, DOUB_INNER, val))
        sectors.append(Sector(theta_min, theta_max, DOUB_INNER, DOUB_OUTER, val * 2))

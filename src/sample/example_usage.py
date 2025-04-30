import os
import time

from src.math.distribution import *
from src.plotting.heatmap import *
from src.plotting.plot_basin import *


"""
This file contains sample code for generating heatmap data using all methods,
as well as finding mu* with basin-hopping and plotting the results.
"""

sigma = 20
n = 300  # grid size
heatmap_filename = '../../heatmap_data/sample.npy'
basin_hopping_filename = '../../basin_data/sample_basin.txt'  # gets overwritten if the file already exists


# Generating and Plotting Heatmap Data by...

# integration (I'm using a lower n because it's really slow)
print('Generating heatmap data by integration...')
heatmap_data_integrate = compute_grid_brute(sigma, calculate_dist_ev_integration, n=5)
plot_heatmap(heatmap_data_integrate,
             sigma=sigma,
             save_fig=True,
             filename='sample_integration.png',
             show=False,
             title='Integration')
print('Saved to images folder.\n')


# monte-carlo
print('Generating heatmap data by monte-carlo...')
heatmap_data_monte_carlo = compute_grid_brute(sigma, calculate_dist_ev_monte_carlo, n=n, mn=1000)
plot_heatmap(heatmap_data_monte_carlo,
             sigma=sigma,
             save_fig=True,
             filename='sample_montecarlo.png',
             show=False,
             title='Monte Carlo')
print('Saved to images folder.\n')


# convolution
print('Generating heatmap data by convolution...')
heatmap_data_convolve = compute_grid_convolve(sigma, n=n)
np.save(heatmap_filename, heatmap_data_convolve)
plot_heatmap(heatmap_data_convolve,
             sigma=sigma,
             save_fig=True,
             filename='sample_convolution.png',
             show=False,
             title='Convolution')
print('Saved to images folder.\n')


# Basin-Hopping. This also will take a while. Lower niter_success for a faster run
if os.path.exists(basin_hopping_filename):
    basin_hopping_filename = f'../../basin_data/sample_{time.time_ns()}.txt'

print('Running basin-hopping...')
find_best_multinormal_center_hopping(
    sigma,
    niter_sucess=3,
    filename=basin_hopping_filename
)

plot_basin_run(basin_hopping_filename, heatmap_filename, save_fig=True, filename='sample_basin_hopping.png')
print('Saved to images folder')








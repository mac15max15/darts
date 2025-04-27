from src.math.distribution import *
from src.plotting.heatmap import *
from src.plotting.plot_basin import *


## Example for Plotting Heatmap and Basin-Hopping ##
sigma = 25
n = 100  # grid size
heatmap_file_name = '../../heatmap_data/sample.npy'
bh_filename = '../../basin_data/sample_basin.txt'


# Generating and Plotting Heatmap Data

# integration (commented out bc it takes forever to run)
# heatmap_data_integrate = compute_grid_brute(sigma, calculate_dist_ev_integration, n=n)

# monte-carlo
heatmap_data_monte_carlo = compute_grid_brute(sigma, calculate_dist_ev_monte_carlo, n=n, mn=1000)

# convolution
heatmap_data_convolve = compute_grid_convolve(sigma, n=n)
np.save(heatmap_file_name, heatmap_data_convolve)

# plot_heatmap(heatmap_data_integrate, sigma=sigma, save_fig=False, show=True, title='Integration')
plot_heatmap(heatmap_data_monte_carlo, sigma=sigma, save_fig=False, show=True, title='Monte Carlo')
plot_heatmap(heatmap_data_convolve, sigma=sigma, save_fig=False, show=True, title='Convolution')


# Uncomment to run basin hopping (also will take a while)
'''
find_best_multinormal_center_hopping(
    sigma,
    t=100/sigma,
    niter_sucess=10,
    filename=bh_filename
)

plot_basin_run(bh_filename, heatmap_file_name)
'''







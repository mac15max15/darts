from src.math.distribution import *
from src.plotting.heatmap import *
from src.plotting.plot_basin import *


## Example for Plotting Heatmap and Basin-Hopping ##
sigma = 10
n = 10  # grid size
heatmap_file_name = '../../heatmap_data/sample.npy'
bh_filename = '../../basin_data/sample_basin.txt'


# Generating and Plotting Heatmap Data

# integration
heatmap_data_integrate = compute_grid_brute(sigma, calculate_dist_ev_integration, n=n)
plot_heatmap(heatmap_data_integrate, sigma=sigma, save_fig=False, show=True, title='Integration')

# convolution
heatmap_data_convolve = compute_grid_convolve(sigma, n=n, save_data=True, filename=heatmap_file_name)
plot_heatmap(heatmap_data_convolve, sigma=sigma, save_fig=False, show=True,title='Convolution')


# Running Basin-Hopping
print('Running Basin Hopping...')
# this will take a while to run. lower niter_sucess for faster but less accurate runs
find_best_multinormal_center_hopping(
    sigma,
    t=100/sigma,
    niter_sucess=3,
    filename=bh_filename
)

plot_basin_run(bh_filename, heatmap_file_name)







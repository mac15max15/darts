from src.math.distribution import *
from src.plotting.heatmap import *
from src.plotting.plot_basin import *


## Example for Plotting Heatmap and Basin-Hopping ##
sigma = 19

# Generating and Plotting Heatmap Data
heatmap_file_name='heatmap_sample.npy'
# heatmap_data = compute_grid_convolve(sigma, n=300, save_data=True, filename=heatmap_file_name)
# plot_heatmap(heatmap_data, save_fig=False, show=False)


# Running Basin-Hopping
bh_filename = 'basin_hopping_sample3.txt'

print('Running Basin Hopping...')
# this will take a while to run. lower niter_sucess for faster but less accurate runs
find_best_multinormal_center_hopping(
    sigma,
    t=100/sigma,
    niter_sucess=5,
    filename=bh_filename
)

plot_basin_run(bh_filename, heatmap_file_name)







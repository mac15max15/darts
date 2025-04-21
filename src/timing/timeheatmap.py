from src.math.distribution import *
import time
import pandas as pd

ns = np.arange(2, 20)
results = np.ndarray((len(ns), 3))

for n in ns:
    t = time.time()

    compute_grid_brute(10, calculate_dist_ev_monte_carlo, n)
    results[n-2, 0] = time.time()-t
    t = time.time()

    compute_grid_brute(10, calculate_dist_ev_monte_carlo, n)
    results[n - 2, 1] = time.time() - t
    t = time.time()

    compute_grid_convolve(10, n=n)
    results[n - 2, 2] = time.time() - t

df = pd.DataFrame(np.column_stack((ns, results)), columns=['n', 'montecarlo10^5', 'montecarlo10^2', 'convolution'])
df.to_csv('heatmap_timing2-200.csv')


from src.math.distribution import *
import time
import pandas as pd

ns = np.arange(2, 100)
results = np.ndarray((len(ns), 4))

for n in ns:
    t = time.time()

    compute_grid_brute(25, calculate_dist_ev_integration, n)
    results[n - 2, 0] = time.time() - t
    t = time.time()

    compute_grid_brute(25, calculate_dist_ev_monte_carlo, n, mn=10**5)
    results[n-2, 1] = time.time()-t
    t = time.time()

    compute_grid_brute(25, calculate_dist_ev_monte_carlo, n, mn=10**2)
    results[n - 2, 2] = time.time() - t
    t = time.time()

    compute_grid_convolve(25, n=n)
    results[n - 2, 3] = time.time() - t


    print(n)

    df = pd.DataFrame(np.column_stack((ns, results)),
                      columns=['n', 'integration', 'montecarlo10^5', 'montecarlo10^2', 'convolution'])
    df.to_csv('heatmap_timing2-100.csv')




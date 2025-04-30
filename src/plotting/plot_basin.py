import matplotlib.pyplot as plt

from src.plotting.display import *
from src.math.constants import *

"""
This file has the code for plotting a basin hopping run.
"""

def parse_file(filename):
    """
    Function for parsing the output file from running a basin hopping optimization.
    Largely written by chatgpt. Enter at your own risk.
    .
    :return: A list of paths, each representing one run of the local optimizer,
    and a list of flags saying whether the result of each path was accepted
    by the basin hopping algo.
    """
    sets = []
    accepted_flags = []

    current_set = []

    with open(filename, 'r') as file:
        previous_line = None
        for line in file:
            line = line.strip()
            if not line:
                continue

            if line.endswith('*') and not previous_line.endswith('*'):
                accepted = line[-5] == 'T'
                accepted_flags.append(accepted)

                if current_set:
                    sets.append(np.array(current_set, dtype=float).transpose())
                    current_set = []
            else:
                # Regular data point
                try:
                    x, y = map(float, line[1:-1].split())
                    current_set.append([x, y])
                except ValueError:
                    print(f"Skipping invalid line: {line}")

            previous_line=line

    return sets, np.array(accepted_flags)


# colormap
def cm(success):
    if success:
        return 'darkgreen'
    else:
        return 'r'


def plot_basin_run(basin_data_path, heatmap_data_path, save_fig=False, filename=None, show=False):
    """
    Take in a file representing a run of the basin hopping algorithm and plot the results.
    :param basin_data_path: path to the text file with the basin-hopping data
    :param heatmap_data_path: path to the .npy file for generating the underlying heatmap (should be the same stdev as used in the basin data)
    """

    fig, ax = generate_dartboard_plot()
    arr = np.load(heatmap_data_path)

    xs = np.linspace(-DOUB_OUTER-HEATMAP_PAD_MM, DOUB_OUTER+HEATMAP_PAD_MM, len(arr))
    ys = np.linspace(-DOUB_OUTER-HEATMAP_PAD_MM, DOUB_OUTER+HEATMAP_PAD_MM, len(arr))

    # plotting the heatmap
    max_pt = np.unravel_index(np.argmax(arr), shape=arr.shape, order='F')
    ax.scatter(xs[max_pt[0]]*SCALE_FACTOR,ys[max_pt[1]]*SCALE_FACTOR, s=100, color='darkgreen', marker='+', linewidths=1.5)
    ax.pcolormesh(
        xs * SCALE_FACTOR, ys * SCALE_FACTOR, arr,
        shading='nearest',
        alpha=0.4,
        cmap=C_MAP
    )

    # actual basin hopping stuff
    local_optimization_runs, accepted_flags = parse_file(basin_data_path)
    num_func_evals = 0

    for i, local_optimization_run in enumerate(local_optimization_runs):
        xs = SCALE_FACTOR*np.array(local_optimization_run[0,:])
        ys = SCALE_FACTOR*np.array(local_optimization_run[1,:])

        num_func_evals += len(xs)

        ax.plot(xs, ys, linestyle='-', marker=None, c=cm(accepted_flags[i]))
        ax.scatter(xs[-1], ys[-1], s=10, c=cm(accepted_flags[i]))
        ax.text(xs[0]+.03, ys[0]+.03, str(i+1), weight='bold')

    fig.tight_layout()
    if save_fig and filename:
        fig.savefig(f'../../images/{filename}', dpi=800)
    print(f'# of F(mu|sigma) evaluations: {num_func_evals}')

    if show:
        plt.show()

if __name__ == "__main__":
    pass

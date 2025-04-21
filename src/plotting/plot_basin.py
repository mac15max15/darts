import matplotlib.pyplot as plt

from src.plotting.display import *
from src.math.constants import *

def parse_file(filename):
    """
    Function for parsing the output file from running a basin hopping optimization.
    Enter at your own risk.
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


def cm(success):
    if success:
        return 'darkgreen'
    else:
        return 'r'


def plot_basin_run(basin_data_path, heatmap_data_path, save_fig=False, output_path=None):

    fig, ax = generate_dartboard_plot()
    arr = np.load(heatmap_data_path)

    xs = np.linspace(-DOUB_OUTER-HEATMAP_PAD_MM, DOUB_OUTER+HEATMAP_PAD_MM, len(arr))
    ys = np.linspace(-DOUB_OUTER-HEATMAP_PAD_MM, DOUB_OUTER+HEATMAP_PAD_MM, len(arr))

    max_pt = np.unravel_index(np.argmax(arr), shape=arr.shape, order='F')

    print(f'Basin-hopping estimated mu*: ({xs[max_pt[0]]:.2f}, {ys[max_pt[1]]:.2f}), ')


    ax.pcolormesh(
        xs * SCALE_FACTOR, ys * SCALE_FACTOR, arr,
        shading='nearest',
        alpha=0.4,
        cmap=C_MAP
    )

    sets, accepted_flags = parse_file(basin_data_path)
    ax.scatter(xs[max_pt[0]]*SCALE_FACTOR,ys[max_pt[1]]*SCALE_FACTOR, s=100, color='darkgreen', marker='+', linewidths=1.5)

    n=0
    for i, s in enumerate(sets):

        xs = SCALE_FACTOR*np.array(s[0,:])
        ys = SCALE_FACTOR*np.array(s[1,:])

        n+=len(xs)

        ax.plot(xs, ys, linestyle='-', marker=None, c=cm(accepted_flags[i]))
        ax.scatter(xs[-1], ys[-1], s=10, c=cm(accepted_flags[i]))
        ax.text(xs[0]+.03, ys[0]+.03, str(i+1), weight='bold')

    fig.tight_layout()
    if save_fig and output_path:
        fig.savefig(output_path, dpi=800)
    print(f'# of F(mu|sigma) evaluations: {n}')
    plt.show()

if __name__ == "__main__":
    plot_basin_run('/Users/maxcaragozian/Desktop/MATH 305/Darts/src/sample/basin_hopping_sample2.txt',
                   '/Users/maxcaragozian/Desktop/MATH 305/Darts/src/sample/heatmap_sample.npy',
                   save_fig=True,
                   output_path='images/sig19basin.png')

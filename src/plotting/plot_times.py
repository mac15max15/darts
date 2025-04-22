import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv('/Users/maxcaragozian/Desktop/MATH 305/Darts/src/timing/heatmap_timing2-100.csv')
    df = df[df['n'] < 20]
    plot_times(df)

def plot_times(timing_df):
    fig, ax = plt.subplots(1, 1, squeeze=True)
    timing_df.plot(
        x='n',
        y=['integration', 'montecarlo10^2','montecarlo10^5','convolution'],
        ax=ax,
        logy=True,
        xticks = np.arange(2, timing_df['n'].max(), step=4)
    )
    ax.legend(loc='lower right', fontsize='small')
    ax.set_ylabel(r'Time to Evaluate $F(\mu \vert \sigma)$ over an n by n grid (seconds)')
    fig.savefig('../../images/timing.png', dpi=800)
    plt.show()




if __name__=='__main__':
    main()
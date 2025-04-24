import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv('../../timing_data/heatmap_timing2-100.csv')
    df = df[df['n'] < 30]
    plot_times(df)

def plot_times(timing_df):
    fig, ax = plt.subplots(1, 1, squeeze=True)

    timing_df.plot(
        x='n',
        y=['integration', 'montecarlo10^2','montecarlo10^5','convolution'],
        ax=ax,
        logy=True,
        xticks = np.arange(0, timing_df['n'].max()+2, step=5),
    )

    ax.legend(loc='lower right',
              fontsize='x-small',
              labels=['Integration', 'Monte-Carlo (100 Samples)', 'Monte-Carlo (10000 Samples)', 'Convolution'],
              title='Method',
              title_fontsize='small')
    ax.set_ylabel(r'Time (seconds)')
    ax.set_ylim((10**-4, 10**3))
    ax.set_xlabel('N')
    ax.set_title(r'Time to Compute $F(\mu \vert \sigma)$ Over an N$\times$N Grid')
    fig.savefig('../../images/timing.png', dpi=800)
    plt.show()




if __name__=='__main__':
    main()
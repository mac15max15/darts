import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv('../../timing_data/heatmap_timing2-20.csv')
    plot_times(df)

def plot_times(timing_df):
    fig, ax = plt.subplots(1, 1, squeeze=True)
    timing_df.plot(
        x='n',
        y=['integration','montecarlo','convolution'],
        ax=ax,
        logy=True,
        xticks = np.arange(2, timing_df['n'].max(), step=4)
    )

    plt.show()




if __name__=='__main__':
    main()
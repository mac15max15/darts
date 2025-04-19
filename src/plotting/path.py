import pandas as pd
from src.plotting.display import *


df = pd.read_csv('big_run.csv')
fig, ax = generate_dartboard_plot()

ax.scatter(SCALE_FACTOR*df['x'], SCALE_FACTOR*df['y'], s=5, c=np.log10(df['ev']), cmap='hot')


fig2, ax2 = plt.subplots(1,1, squeeze=True)
ax2.plot(df.index, df['ev'])
plt.show()






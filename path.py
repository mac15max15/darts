import matplotlib.pyplot as plt
import pandas as pd
from display import *


df = pd.read_csv('results/26-37.csv')

fig, ax = generate_dartboard_plot()

ax.scatter(SCALE_FACTOR*df['x'], SCALE_FACTOR*df['y'], s=5, c=df['ev'])


fig2, ax2 = plt.subplots(1,1, squeeze=True)
ax2.plot(df['sigma'], df['ev'])
plt.show()






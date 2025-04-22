import pandas as pd
from src.plotting.display import *


df = pd.read_csv('/Users/maxcaragozian/Desktop/MATH 305/Darts/src/math/centers1-100_500x500.csv')
fig, ax = generate_dartboard_plot()

scatter = ax.scatter(SCALE_FACTOR*df['x'], SCALE_FACTOR*df['y'], s=5, c=df['sigma'], cmap='viridis')
fig.colorbar(scatter, label='$\sigma$ (mm)', location='bottom', aspect=40, use_gridspec=True, shrink=0.5, pad=0.01)
ax.set_title('$\mu^*$ Paramaterized By $\sigma$')
fig.tight_layout()
fig.savefig('../../images/mustar.png')

plt.show()







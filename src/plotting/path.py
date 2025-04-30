import pandas as pd
from src.plotting.display import *

"""
This generates a graphic I didn't end up using in the report. It's not really usable, but I'm keeping it here
in case I ever decide to fix it up.
"""

df = pd.read_csv('/Users/maxcaragozian/Desktop/MATH 305/Darts/src/math/centers1-100_500x500.csv')
fig, ax = generate_dartboard_plot()

scatter = ax.scatter(SCALE_FACTOR*df['x'], SCALE_FACTOR*df['y'], s=5, c=df['sigma'], cmap='viridis')
fig.colorbar(scatter, label='$\sigma$ (mm)', location='bottom', aspect=30, use_gridspec=True, shrink=0.5, pad=0.02)
#ax.set_title('$\mu^*$ Paramaterized By $\sigma$')
fig.tight_layout()
#fig.savefig('../../images/mustar2.png', dpi=DPI)

plt.show()







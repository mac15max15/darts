from src.plotting.display import *
from src.math.constants import *

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

fname = 'high_res_26.9.npy'
arr = np.load(f'../../heatmap_data/{fname}')

xs = np.linspace(-DOUB_OUTER-HEATMAP_PAD_MM, DOUB_OUTER+HEATMAP_PAD_MM, len(arr))
ys = np.linspace(-DOUB_OUTER-HEATMAP_PAD_MM, DOUB_OUTER+HEATMAP_PAD_MM, len(arr))

X, Y = np.meshgrid(xs, ys)

ax.set_xlabel('mu_x')
ax.set_ylabel('mu_y')
ax.set_zlabel('Expected Score')

ax.plot_surface(X, Y, -arr, cmap='coolwarm')
plt.show()
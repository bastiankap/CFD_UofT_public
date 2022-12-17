import matplotlib.pyplot as plt
import numpy as np

def plot_contour(x_nodes, y_nodes,array, title):
    X, Y = np.meshgrid(x_nodes, y_nodes)
    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
    plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
    plt.pcolormesh(X, Y, array, cmap='hot')
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.show()

def plot_streamline(x_nodes,y_nodes,u_field,v_field, Re, nx):
    X, Y = np.meshgrid(x_nodes, y_nodes[::-1])
    v_flip = np.flip(v_field,0)
    u_flip = np.flip(u_field,0)
    plt.streamplot(X, Y, u_flip, v_flip, density=2)
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.title(f'Re: {Re}, nx: {nx}')
    plt.savefig(f'Figures/Re{Re}_nx{nx}.png')
    plt.show()
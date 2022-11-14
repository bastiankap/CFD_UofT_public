import numpy as np
from scipy.sparse import dok_matrix
from scipy import interpolate
import time
from solver import solve_cgs  # you will have to import your own solver
import matplotlib.pyplot as plt
from math import log
# Hi Mehdi, thank you so much again for your help throughout this project :)

# Define parameters
gamma = 20
h_f = 10
phi_x0 = 10
phi_xL = 100
phi_ext = 300
L_x = 1
L_y = 1
# n_x and n_y for coarse, fine and extra fine mesh
n_c = 80
n_f = 160
n_ff = 320
r_i = 1.01

phi_collection = {}  # dict to store the results for all combinations of n and r
x_center_collection = {}  # dict to store the x location of the nodes for different meshes
y_center_collection = {}  # dict to store the y location of the nodes for different meshes
for n_iter in [n_c,n_f,n_ff]:
    for r_iter in [1,r_i]:
        nx = n_iter
        ny = n_iter

        # generate mesh
        r = r_iter  # inflation factor
        if r > 1:  # generate inflated mesh
            print('Starting non-uniform mesh solver...')
            x_0 = ((1 - r) / (1 - r ** (nx / 2))) * L_x / 2  # boundary cell
            y_0 = ((1 - r) / (1 - r ** (ny / 2))) * L_y / 2  # boundary cell
            width_x = [x_0]  # list of all x-widths from left to right
            width_y = [y_0]  # list of all y-widths from top to bottom

            while round(sum(width_x), ndigits=5) < L_x / 2:
                x = width_x[-1] * r
                width_x.append(x)
            width_x = width_x + list(reversed(width_x))

            while round(sum(width_y), ndigits=5) < L_y / 2:
                y = width_y[-1] * r
                width_y.append(y)
            width_y = width_y + list(reversed(width_y))

            delta_x = [x_0]  # list of all distances between neighbouring nodes from left to right
            delta_y = [y_0]  # list of all distances between neighbouring nodes from top to bottom

            for i in np.arange(0, len(width_x) - 1):
                delta_x.append(width_x[i] / 2 + width_x[i + 1] / 2)
            delta_x.append(x_0)
            for i in np.arange(0, len(width_y) - 1):
                delta_y.append(width_y[i] / 2 + width_y[i + 1] / 2)
            delta_y.append(y_0)

        else:  # generate uniform mesh if r = 1
            print('Starting uniform mesh solver...')
            width_x = np.ones(nx + 1) * (L_x / nx)
            width_y = np.ones(ny + 1) * (L_y / ny)
            delta_x = width_x
            delta_y = width_y
            x_0 = width_x[0]
            y_0 = width_y[0]

        a_w = np.zeros(shape=(ny, nx))
        a_e = np.zeros(shape=(ny, nx))
        a_n = np.zeros(shape=(ny, nx))
        a_s = np.zeros(shape=(ny, nx))
        a_p = np.zeros(shape=(ny, nx))
        # generate non-boundary values for a_s,a_n,a_w,a_e
        for i in np.arange(0, ny - 1):
            for j in np.arange(0, nx):
                a_s[i, j] = gamma * width_x[j] / delta_y[i + 1]

        for i in np.arange(1, ny):
            for j in np.arange(0, nx):
                a_n[i, j] = gamma * width_x[j] / delta_y[i]

        for i in np.arange(0, ny):
            for j in np.arange(1, nx):
                a_w[i, j] = gamma * width_y[i] / delta_x[j]

        for i in np.arange(0, ny):
            for j in np.arange(0, nx - 1):
                a_e[i, j] = gamma * width_y[i] / delta_x[j + 1]

        # generate source terms for the boundaries
        s_u = np.zeros(shape=(ny, nx))
        s_p = np.zeros(shape=(ny, nx))

        # source terms on upper boundary (i = 0)
        i = 0
        for j in np.arange(0, nx):
            s_u[i, j] = s_u[i, j] + (2 * width_x[j] / delta_y[i]) * gamma * (delta_y[i] * h_f * phi_ext) / \
                        (2 * gamma + delta_y[i] * h_f)
            s_p[i, j] = s_p[i, j] + (2 * width_x[j] / delta_y[i]) * gamma * \
                        (2 * gamma / (2 * gamma + delta_y[i] * h_f) - 1)
        # source terms for lower boundary (i = ny-1)
        i = ny - 1
        for j in np.arange(0, nx):
            s_u[i, j] = s_u[i, j] + 0
            s_p[i, j] = s_p[i, j] + 0

        # source terms for left boundary (j = j)
        j = 0
        for i in np.arange(0, ny):
            s_u[i, j] = s_u[i, j] + 2 * gamma * width_y[i] / delta_x[j] * phi_x0
            s_p[i, j] = s_p[i, j] + -2 * gamma * width_y[i] / delta_x[j]

        # source terms for right boundary (j = nx-1)
        j = nx - 1
        for i in np.arange(0, ny):
            s_u[i, j] = s_u[i, j] + 2 * gamma * width_y[i] / delta_x[j + 1] * phi_xL
            s_p[i, j] = s_p[i, j] + -2 * gamma * width_y[i] / delta_x[j + 1]

        # generate a_p for all nodes
        for i in np.arange(0, ny):
            for j in np.arange(0, nx):
                a_p[i, j] = a_e[i, j] + a_w[i, j] + a_s[i, j] + a_n[i, j] - s_p[i, j]

        # build matrix A and vector b for all nodes
        A = dok_matrix((nx * ny, nx * ny))
        b = dok_matrix((nx * ny, 1))

        for i in np.arange(0, ny):
            for j in np.arange(0, nx):

                k = (i * nx + j)

                A[k, k] = a_p[i, j]
                if k + 1 <= (nx * ny) - 1:
                    A[k, k + 1] = -a_e[i, j]
                if k - 1 >= 0:
                    A[k, k - 1] = -a_w[i, j]
                if k + nx <= (nx * ny) - 1:
                    A[k, k + nx] = -a_s[i, j]
                if k - nx >= 0:
                    A[k, k - nx] = -a_n[i, j]
                b[k] = s_u[i, j]

        phi, err, t_solve = solve_cgs(A, b.toarray(), tol=1e-10)  # solve A * phi = b using my own solver
        print(f'L2-residual with n = {n_iter}, r = {r}: {np.linalg.norm(err)}')
        print(f'max temp: {max(phi)}')
        print(f'min temp: {min(phi)}')

        # visualize
        phi_array_list = []  # convert vector phi into a 2D array

        for i in np.arange(0, ny):
            phi_array_list.append(phi[i * ny:(i * ny + nx)])

        phi_array = np.asarray(phi_array_list)

        phi_collection[f'n{n_iter},r{r}'] = phi_array

        # create non-uniform heatmap
        x_center = [x_0 / 2]
        for j in np.arange(1, len(delta_x) - 1):
            x_center.append(x_center[-1] + delta_x[j])

        x_center_collection[f'n{n_iter},r{r}'] = x_center

        y_center = [y_0 / 2]
        for i in np.arange(1, len(delta_y) - 1):
            y_center.append(y_center[-1] + delta_y[i])

        y_center_collection[f'n{n_iter},r{r}'] = y_center

        X, Y = np.meshgrid(x_center, y_center)

        plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
        plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
        plt.pcolormesh(X, Y, phi_array, cmap='hot')
        plt.gca().invert_yaxis()
        plt.colorbar()
        plt.title(f"nx = ny = {n_iter}, r = {round(r, ndigits=2)}", fontsize=15)
        plt.tight_layout()
        plt.savefig(f'Figures/n{n_iter}r{r_iter}.png')
        plt.show()

# # calculate convergence

print("beginning interpolation")
for r in [1, r_i]:
    # load coordinates of nodes for coarse, fine and extra fine mesh
    x_coarse = x_center_collection[f'n{n_c},r{r}']
    y_coarse = y_center_collection[f'n{n_c},r{r}']
    phi_coarse = phi_collection[f'n{n_c},r{r}']

    x_fine = x_center_collection[f'n{n_f},r{r}']
    y_fine = y_center_collection[f'n{n_f},r{r}']
    phi_fine = phi_collection[f'n{n_f},r{r}']

    x_finest = x_center_collection[f'n{n_ff},r{r}']
    y_finest = y_center_collection[f'n{n_ff},r{r}']
    phi_finest = phi_collection[f'n{n_ff},r{r}']

    # interpolate values for coarse and fine mesh
    f_coarse = interpolate.interp2d(x_coarse, y_coarse, phi_coarse, kind='cubic')
    f_fine = interpolate.interp2d(x_fine, y_fine, phi_fine, kind='cubic')

    e_coarse = 0
    e_fine = 0

    # calculate deviation of coarse and fine mesh from extra fine mesh at every node of the extra fine mesh
    for j in np.arange(len(x_center_collection[f'n{n_ff},r{r}'])):
        for i in np.arange(len(y_center_collection[f'n{n_ff},r{r}'])):
            e_coarse = e_coarse + pow((phi_finest[i, j]
                                       - f_coarse(x_center_collection[f'n{n_ff},r{r}'][j],
                                                  y_center_collection[f'n{n_ff},r{r}'][i])), 2)
            e_fine = e_fine + pow((phi_finest[i, j]
                                   - f_fine(x_center_collection[f'n{n_ff},r{r}'][j],
                                            y_center_collection[f'n{n_ff},r{r}'][i])), 2)

    e_fine = np.sqrt(e_fine / (len(x_center_collection[f'n{n_ff},r{r}']) * len(y_center_collection[f'n{n_ff},r{r}'])))
    e_coarse = np.sqrt(
        e_coarse / (len(x_center_collection[f'n{n_ff},r{r}']) * len(y_center_collection[f'n{n_ff},r{r}'])))

    o_conv = log(e_coarse / e_fine) / (log((L_x / n_c) / (L_x / n_f)))
    print(f'For r = {r} the order of convergence is: {o_conv}')
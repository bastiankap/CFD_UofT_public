import numpy as np
from solver import solve_cgs
from scipy.sparse import dok_matrix
import matplotlib.pyplot as plt
from scipy import interpolate
from math import log
import time

def generate_grid(nx=10, ny=10, L_x=1, L_y=1):  # generate nodal positions and cell width/height
    dx = L_x / nx
    dy = L_y / ny
    x_node = [dx / 2]
    y_node = [L_y - dy / 2]  # started enumerating from top left, but coordinate frame is in bottom left!

    for i in np.arange(ny - 1):
        y_node.append(y_node[-1] - dy)
    for j in np.arange(nx - 1):
        x_node.append(x_node[-1] + dx)
    x_node = np.asarray(x_node)
    y_node = np.asarray(y_node)
    return dx, dy, x_node, y_node


def generate_velocities(case='constant', x=0, y=0, dx=0.1, dy=0.1, L_x=1, L_y=1):  # generate velocities at each face
    if case == 'constant':
        u_e = -2000
        u_w = -2000
        u_n = -2000
        u_s = -2000

    elif case == 'circular':
        x_w = (x - dx / 2)
        y_w = y
        x_e = (x + dx / 2)
        y_e = y
        x_n = x
        y_n = (y + dy / 2)
        x_s = x
        y_s = (y - dy / 2)

        r_e = np.sqrt(pow((x_e - L_x / 2), 2) + pow((y_e - L_y / 2), 2))
        r_w = np.sqrt(pow((x_w - L_x / 2), 2) + pow((y_w - L_y / 2), 2))
        r_n = np.sqrt(pow((x_n - L_x / 2), 2) + pow((y_n - L_y / 2), 2))
        r_s = np.sqrt(pow((x_s - L_x / 2), 2) + pow((y_s - L_y / 2), 2))

        theta_e = np.arctan2(y_e - L_y / 2, x_e - L_x / 2)
        theta_w = np.arctan2(y_w - L_y / 2, x_w - L_x / 2)
        theta_n = np.arctan2(y_n - L_y / 2, x_n - L_x / 2)
        theta_s = np.arctan2(y_s - L_y / 2, x_s - L_x / 2)

        u_e = -r_e * np.sin(theta_e)
        u_w = -r_w * np.sin(theta_w)
        u_n = r_n * np.cos(theta_n)
        u_s = r_s * np.cos(theta_s)
    else:
        raise Exception('chose "constant" or "circular"')  # raise error if wrong input is chosen

    return u_e, u_w, u_n, u_s


def generate_DF(u_e=1, u_w=1, u_n=1, u_s=1, dx=1, dy=1, gamma=5, rho=1):  # generate D and F values for each face
    D_e = gamma / dx
    D_w = gamma / dx
    D_n = gamma / dy
    D_s = gamma / dy
    F_e = rho * u_e
    F_w = rho * u_w
    F_n = rho * u_n
    F_s = rho * u_s

    return D_e, D_w, D_n, D_s, F_e, F_w, F_n, F_s


def generate_a_coefficients(D_e=1, D_w=1, D_n=1, D_s=1, F_e=1, F_w=1, F_n=1, F_s=1, scheme='central'):  # generate
    # a-coefficients
    if scheme == 'central':
        # print('using central differencing')
        Pe = max(abs(F_e / D_e), abs(F_w / D_w), abs(F_n / D_n), abs(F_s / D_s))
        if Pe >= 2:  # check for peclet number
            raise Exception('Pe > 2! use upwind instead')

        a_e = (D_e - F_e / 2)
        a_w = (D_w + F_w / 2)
        a_n = (D_n - F_n / 2)
        a_s = (D_s + F_s / 2)

    elif scheme == 'upwind':
        # print('using upwind')

        a_e = D_e + max(0, -F_e)
        a_w = D_w + max(F_w, 0)
        a_n = D_n + max(0, -F_n)
        a_s = D_s + max(F_s, 0)

    else:
        raise Exception('chose either "central" or "upwind"')

    return a_e, a_w, a_n, a_s


def generate_source_terms(i, j, D_e=1, D_w=1, D_n=1, D_s=1, F_e=1, F_w=1, F_n=1, F_s=1, nx=10, ny=10, scheme='central'):
    # generate source terms if node is located at a boundary
    s_u = 0
    s_p = 0
    phi_b_n = 100
    phi_b_w = 100
    phi_b_e = 0
    phi_b_s = 0

    if i == 0:

        if scheme == 'central':
            # print('central')
            s_u = s_u + (2 * D_n - F_n) * phi_b_n
            s_p = s_p + (-(2 * D_n - F_n))
        elif scheme == 'upwind':
            # print('upwind')
            s_u = s_u + (2 * D_n + max(0, -F_n)) * phi_b_n
            s_p = s_p + (-(2 * D_n + max(0, -F_n)))

    if i == ny - 1:

        if scheme == 'central':
            # print('central')
            s_u = s_u + (2 * D_s + F_s) * phi_b_s
            s_p = s_p + (-(2 * D_s + F_s))
        elif scheme == 'upwind':
            # print('upwind')
            s_u = s_u + (2 * D_s + max(F_s, 0)) * phi_b_s
            s_p = s_p + (-(2 * D_s + max(F_s, 0)))

    if j == 0:

        if scheme == 'central':
            # print('central')
            s_u = s_u + (2 * D_w + F_w) * phi_b_w
            s_p = s_p + (-(2 * D_w + F_w))
        elif scheme == 'upwind':
            # print('upwind')
            s_u = s_u + (2 * D_w + max(F_w, 0)) * phi_b_w
            s_p = s_p + (-(2 * D_w + max(F_w, 0)))

    if j == nx - 1:

        if scheme == 'central':
            # print('central')
            s_u = s_u + (2 * D_e - F_e) * phi_b_e
            s_p = s_p + (-(2 * D_e - F_e))
        elif scheme == 'upwind':
            # print('upwind')
            s_u = s_u + (2 * D_e + max(0, -F_e)) * phi_b_e
            s_p = s_p + (-(2 * D_e + max(0, -F_e)))

    return s_u, s_p


def delete_a_coefficients(i, j, a_e, a_w, a_n, a_s, nx=10, ny=10):
    # delete a-coefficients of faces that lay on a boundary
    a_e = a_e
    a_w = a_w
    a_s = a_s
    a_n = a_n
    if i == 0:
        a_n = 0
    if i == ny - 1:
        a_s = 0
    if j == 0:
        a_w = 0
    if j == nx - 1:
        a_e = 0
    return a_e, a_w, a_n, a_s


def build(nx=10, ny=10, x_n=np.zeros(500), y_n=np.zeros(500), dx=1, dy=1, scheme='central', case='constant', gamma=5,
          rho=1):  # build A matrix and b vector
    A = dok_matrix((nx * ny, nx * ny))
    b = dok_matrix((nx * ny, 1))

    for i in np.arange(0, ny):
        for j in np.arange(0, nx):
            y_iter = y_n[i]
            x_iter = x_n[j]
            u_e, u_w, u_n, u_s = generate_velocities(case=case, x=x_iter, y=y_iter, dx=dx, dy=dy)
            D_e, D_w, D_n, D_s, F_e, F_w, F_n, F_s = generate_DF(u_e, u_w, u_n, u_s, dx=dx, dy=dy, gamma=gamma, rho=rho)
            a_e, a_w, a_n, a_s = generate_a_coefficients(D_e=D_e, D_w=D_w, D_n=D_n, D_s=D_s, F_e=F_e, F_w=F_w, F_n=F_n,
                                                         F_s=F_s, scheme=scheme)
            s_u, s_p = generate_source_terms(i, j, D_e, D_w, D_n, D_s, F_e, F_w, F_n, F_s, nx=nx, ny=ny, scheme=scheme)
            a_e, a_w, a_n, a_s = delete_a_coefficients(i, j, a_e, a_w, a_n, a_s, nx, ny)

            a_p = a_e + a_w + a_n + a_s + (F_e - F_w + F_n - F_s) - s_p

            # assemble A matrix and b-vector
            k = (i * nx + j)

            A[k, k] = a_p
            if k + 1 <= (nx * ny) - 1:
                A[k, k + 1] = -a_e
            if k - 1 >= 0:
                A[k, k - 1] = -a_w
            if k + nx <= (nx * ny) - 1:
                A[k, k + nx] = -a_s
            if k - nx >= 0:
                A[k, k - nx] = -a_n
            b[k] = s_u
    b = b.toarray()
    return A, b


def run_analysis(nx=10, ny=10, scheme='central', case='constant', L_x=1, L_y=1, gamma=5, rho=1, plot=True):
    dx, dy, x_node, y_node = generate_grid(nx=nx, ny=ny, L_x=L_x, L_y=L_y)
    A, b = build(nx=nx, ny=ny, x_n=x_node, y_n=y_node, dx=dx, dy=dy, scheme=scheme, case=case, gamma=gamma, rho=rho)
    phi, err, time = solve_cgs(A, b)

    phi_array_list = []  # convert vector phi into a 2D array

    for i in np.arange(0, ny):
        phi_array_list.append(phi[i * ny:(i * ny + nx)])

    phi_array = np.asarray(phi_array_list)

    if plot:
        X, Y = np.meshgrid(x_node, y_node[::-1])
        plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
        plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
        plt.pcolormesh(X, Y, phi_array, cmap='hot')
        plt.gca().invert_yaxis()
        plt.colorbar()
        plt.title(f"nx = ny = {nx}, scheme: {scheme}", fontsize=15)
        plt.tight_layout()
        plt.savefig(f'Figures/n{nx}_{scheme}_{case}_g{gamma}.png')
        plt.show()

    print(f'max: {np.max(phi_array)}')
    print(f'min: {np.min(phi_array)}')

    return phi_array, err, time, x_node, y_node


# Question 2:
def question_2():  # showcase false diffusion introduced by upwind
    plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = False
    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = True
    phi_10, _, _, x_10, y_10 = run_analysis(nx=10, ny=10, scheme='upwind', case='constant', L_x=1, L_y=1, gamma=0,
                                            rho=1, plot=False)
    phi_50, _, _, x_50, y_50 = run_analysis(nx=50, ny=50, scheme='upwind', case='constant', L_x=1, L_y=1, gamma=0,
                                            rho=1, plot=False)
    phi_100, _, _, x_100, y_100 = run_analysis(nx=100, ny=100, scheme='upwind', case='constant', L_x=1, L_y=1, gamma=0,
                                               rho=1, plot=False)
    x_plot_10 = np.sqrt(x_10 ** 2 + y_10[::-1] ** 2)
    x_plot_50 = np.sqrt(x_50 ** 2 + y_50[::-1] ** 2)
    x_plot_100 = np.sqrt(x_100 ** 2 + y_100[::-1] ** 2)
    plt.plot(x_plot_10, phi_10.diagonal(), '--k', label='upwind 10x10')
    plt.plot(x_plot_50, phi_50.diagonal(), '-.k', label='upwind 50x50')
    plt.plot(x_plot_100, phi_100.diagonal(), '-k', label='upwind 100x100')
    plt.axvline(x=np.sqrt(2) / 2, color='red', label='exact solution')
    plt.legend()
    plt.xlabel('Distance along Diagonal')
    plt.ylabel('phi')
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig('Figures/question_2.png')
    plt.show()


# Question 4:
def question_4(plot=True, buffer=0, gamma=5):  # calculate order of convergence
    """
    this function calculates the order of convergence using upwind and central differencing
    :param plot: bool that activates/deactivates plotting for each run
    :param buffer: layers to the boundary that are left out for error calculation
    :param gamma: diffusive parameter
    :return:
    """
    O = {}
    for scheme in ['upwind', 'central']:
        print(f'Calculating Order of Convergence for {scheme}')
        phi_coarse, _, _, x_coarse, y_coarse = run_analysis(nx=80, ny=80, scheme=scheme, case='circular', L_x=1, L_y=1,
                                                            gamma=gamma, rho=1, plot=plot)
        phi_fine, _, _, x_fine, y_fine = run_analysis(nx=160, ny=160, scheme=scheme, case='circular', L_x=1, L_y=1,
                                                      gamma=gamma, rho=1, plot=plot)
        phi_finest, _, _, x_finest, y_finest = run_analysis(nx=320, ny=320, scheme=scheme, case='circular', L_x=1,
                                                            L_y=1, gamma=gamma, rho=1, plot=plot)

        f_coarse = interpolate.interp2d(x_coarse, y_coarse, phi_coarse, kind='cubic')
        f_fine = interpolate.interp2d(x_fine, y_fine, phi_fine, kind='cubic')

        e_coarse = 0
        e_fine = 0
        for j in np.arange(buffer, len(x_finest) - buffer):
            for i in np.arange(buffer, len(y_finest) - buffer):
                e_coarse = e_coarse + pow((phi_finest[i, j] - f_coarse(x_finest[j], y_finest[i])), 2)
                e_fine = e_fine + pow((phi_finest[i, j] - f_fine(x_finest[j], y_finest[i])), 2)

        e_fine = np.sqrt(e_fine / (len(x_finest) * len(y_finest)))
        e_coarse = np.sqrt(e_coarse / (len(x_finest) * len(y_finest)))
        o_conv = log(e_coarse / e_fine) / (log((1 / 160) / (1 / 320)))

        O[scheme] = o_conv

    print(f'Upwind O:{O["upwind"]}')
    print(f'Central O:{O["central"]}')


#question_2()
#question_4(plot=False, buffer=0, gamma=5)
# run_analysis(nx=80, ny=80, scheme='upwind', case='circular', L_x=1, L_y=1,gamma=5, rho=1, plot=True)
# phi_array, err, time, x_node, y_node = run_analysis(nx=320, ny=320, scheme='upwind', case='circular', L_x=1, L_y=1,
                                                   # gamma=5, rho=1, plot=True)
# run_analysis(nx=320, ny=320, scheme='upwind', case='constant', L_x=1, L_y=1, gamma=5, rho=1, plot=True)
run_analysis(nx=320, ny=320, scheme='central', case='constant', L_x=1, L_y=1, gamma=5, rho=1, plot=True)

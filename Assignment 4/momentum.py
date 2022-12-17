from scipy.sparse import dok_matrix
import numpy as np
from solver import solve_cgs
import matplotlib.pyplot as plt


def gen_FD(i, j, u, v, p, x_ap, y_ap, dx, dy, Re, nx, ny, alpha_f):
    # Generate F-Coefficients
    # velocities of node and neighbouring nodes
    u_P = u[i, j]
    v_P = v[i, j]
    u_E = u[i, j + 1]
    u_W = u[i, j - 1]
    v_N = v[i - 1, j]
    v_S = v[i + 1, j]
    # pressures of node and direct neighbours
    p_P = p[i, j]
    p_E = p[i, j + 1]
    p_W = p[i, j - 1]
    p_N = p[i - 1, j]
    p_S = p[i + 1, j]

    # pressures of second neighbours (only for nodes that are not at the boundary ( add if condition)
    n_rows, n_cols = p.shape
    if j < nx:
        p_EE = p[i, j + 2]
    if j - 2 >= 0:
        p_WW = p[i, j - 2]
    if i - 2 >= 0:
        p_NN = p[i - 2, j]
    if i < ny:
        p_SS = p[i + 2, j]

    # pressure gradients for u momentum
    u_dp_dx_P = (1 / 2 * (p_E + p_P) - 1 / 2 * (p_W + p_P)) / dx
    u_dp_dx_ef = (p_E - p_P) / dx  # ef: east face
    if j + 2 < n_cols:
        u_dp_dx_E = (1 / 2 * (p_EE + p_E) - 1 / 2 * (p_E + p_P)) / dx
    u_dp_dx_wf = (p_P - p_W) / dx
    if j - 2 >= 0:
        u_dp_dx_W = (1 / 2 * (p_P + p_W) - 1 / 2 * (p_WW + p_W)) / dx

    # pressure gradients for v momentum
    v_dp_dy_P = (1 / 2 * (p_N + p_P) - 1 / 2 * (p_S + p_P)) / dy
    v_dp_dy_nf = (p_N - p_P) / dy
    if i - 2 >= 0:
        v_dp_dy_N = (1 / 2 * (p_NN + p_N) - 1 / 2 * (p_N + p_P)) / dy
    v_dp_dy_sf = (p_P - p_S) / dy
    if i + 2 < n_rows:
        v_dp_dy_S = (1 / 2 * (p_P + p_S) - 1 / 2 * (p_S + p_SS)) / dy

    # generate ap of x and y momentum
    x_ap_P = x_ap[i, j]
    x_ap_E = x_ap[i, j + 1]
    x_ap_W = x_ap[i, j - 1]
    x_ap_ef = (x_ap_P + x_ap_E) / 2
    x_ap_wf = (x_ap_P + x_ap_W) / 2
    y_ap_P = y_ap[i, j]
    y_ap_N = y_ap[i - 1, j]
    y_ap_S = y_ap[i + 1, j]
    y_ap_nf = (y_ap_N + y_ap_P) / 2
    y_ap_sf = (y_ap_S + y_ap_P) / 2

    V = dx

    # calculate velocities at faces of control volume
    if j == n_cols - 2:
        u_ef = 0
    else:
        u_ef = (u_E + u_P) / 2 + 1 / 2 * alpha_f * ((V / x_ap_E) * u_dp_dx_E + (V / x_ap_P) * u_dp_dx_P) - alpha_f * (
                V / x_ap_ef) * u_dp_dx_ef

    if j == 1:
        u_wf = 0
    else:
        u_wf = (u_W + u_P) / 2 + 1 / 2 * alpha_f * ((V / x_ap_W) * u_dp_dx_W + (V / x_ap_P) * u_dp_dx_P) - alpha_f * (
                V / x_ap_wf) * u_dp_dx_wf

    if i == 1:
        v_nf = 0
    else:
        v_nf = (v_N + v_P) / 2 + 1 / 2 * alpha_f * ((V / y_ap_N) * v_dp_dy_N + (V / y_ap_P) * v_dp_dy_P) - alpha_f * (
                V / y_ap_nf) * v_dp_dy_nf

    if i == n_rows - 2:
        v_sf = 0
    else:
        v_sf = (v_S + v_P) / 2 + 1 / 2 * alpha_f * ((V / y_ap_S) * v_dp_dy_S + (V / y_ap_P) * v_dp_dy_P) - alpha_f * (
                V / y_ap_sf) * v_dp_dy_sf

    F_e = u_ef
    F_w = u_wf
    F_n = v_nf
    F_s = v_sf

    # Calculating D values
    D_e = 1 / (Re * dx)
    D_w = 1 / (Re * dx)
    D_n = 1 / (Re * dy)
    D_s = 1 / (Re * dy)

    return F_e, F_w, F_n, F_s, D_e, D_w, D_n, D_s


def gen_a(F_e, F_w, F_n, F_s, D_e, D_w, D_n, D_s):
    a_e = (D_e - F_e / 2)
    a_w = (D_w + F_w / 2)
    a_n = (D_n - F_n / 2)
    a_s = (D_s + F_s / 2)
    return a_e, a_w, a_n, a_s


def gen_source(i, j, p, F_e, F_w, F_n, F_s, D_e, D_w, D_n, D_s, nx, ny, dx, dy, direction='x'):
    # generate source terms if node is located at a boundary
    s_u = 0
    s_p = 0

    if direction == 'x':
        north_value = 1
        p_P = p[i, j]
        p_E = p[i, j + 1]
        p_W = p[i, j - 1]
        p_ef = 1 / 2 * (p_P + p_E)
        p_wf = 1 / 2 * (p_P + p_W)
        s_u = s_u + (p_wf - p_ef)
    elif direction == 'y':
        north_value = 0
        p_P = p[i, j]
        p_N = p[i - 1, j]
        p_S = p[i + 1, j]
        p_nf = 1 / 2 * (p_P + p_N)
        p_sf = 1 / 2 * (p_P + p_S)
        s_u = s_u + (p_sf - p_nf)
    east_value = 0
    west_value = 0
    south_value = 0

    if i == 1:
        s_u = s_u + (2 * D_n - F_n) * north_value
        s_p = s_p + (-(2 * D_n - F_n))
    if i == ny:
        s_u = s_u + (2 * D_s + F_s) * south_value
        s_p = s_p + (-(2 * D_s + F_s))
    if j == 1:
        s_u = s_u + (2 * D_w + F_w) * west_value
        s_p = s_p + (-(2 * D_w + F_w))
    if j == nx:
        s_u = s_u + (2 * D_e - F_e) * east_value
        s_p = s_p + (-(2 * D_e - F_e))

    return s_u, s_p


def del_a(i, j, a_e, a_w, a_n, a_s, nx, ny):
    # delete a-coefficients of faces that lay on a boundary
    a_e = a_e
    a_w = a_w
    a_s = a_s
    a_n = a_n
    if i == 1:
        a_n = 0
    if i == ny:
        a_s = 0
    if j == 1:
        a_w = 0
    if j == nx:
        a_e = 0
    return a_e, a_w, a_n, a_s


def build_Ab(u, v, p, x_ap, y_ap, Re, nx, ny, alpha_uv_solver, alpha_f, direction='x', assem=True):
    A = dok_matrix((nx * ny, nx * ny))
    b = dok_matrix((nx * ny, 1))
    F_e_array = np.zeros((nx + 2, ny + 2))
    F_w_array = np.zeros((nx + 2, ny + 2))
    F_n_array = np.zeros((nx + 2, ny + 2))
    F_s_array = np.zeros((nx + 2, ny + 2))

    if direction == 'x':
        a_p_return = np.copy(x_ap)
    elif direction == 'y':
        a_p_return = np.copy(y_ap)
    else:
        raise ValueError('direction must be either x or y')

    for i in np.arange(0, ny):
        for j in np.arange(0, nx):
            F_e, F_w, F_n, F_s, D_e, D_w, D_n, D_s = gen_FD(i + 1, j + 1, u=u, v=v, p=p, x_ap=x_ap, y_ap=y_ap, Re=Re,
                                                            dx=1 / nx, dy=1 / ny, nx=nx, ny=ny, alpha_f=alpha_f)
            a_e, a_w, a_n, a_s = gen_a(F_e, F_w, F_n, F_s, D_e, D_w, D_n, D_s)
            a_e, a_w, a_n, a_s = del_a(i + 1, j + 1, a_e, a_w, a_n, a_s, nx=nx, ny=ny)
            s_u, s_p = gen_source(i + 1, j + 1, p=p, direction=direction, F_e=F_e, F_w=F_w, F_n=F_n, F_s=F_s, D_e=D_e,
                                  D_w=D_w, D_n=D_n, D_s=D_s,
                                  nx=nx, ny=ny, dx=1 / nx, dy=1 / ny)
            a_p = a_e + a_w + a_n + a_s + (F_e - F_w + F_n - F_s) - s_p

            a_p_return[i + 1, j + 1] = a_p


            if assem == True:
                k = (i * nx + j)
                A[k, k] = a_p/alpha_uv_solver

                if k + 1 <= (nx * ny) - 1:
                    A[k, k + 1] = -a_e
                if k - 1 >= 0:
                    A[k, k - 1] = -a_w
                if k + nx <= (nx * ny) - 1:
                    A[k, k + nx] = -a_s
                if k - nx >= 0:
                    A[k, k - nx] = -a_n
                if direction == 'x':
                    b[k] = s_u + (1-alpha_uv_solver)*a_p/alpha_uv_solver * u[i+1, j+1]
                else:
                    b[k] = s_u + + (1-alpha_uv_solver)*a_p/alpha_uv_solver * v[i+1, j+1]

            elif assem == False:
                F_e_array[i + 1, j + 1] = F_e
                F_w_array[i + 1, j + 1] = F_w
                F_n_array[i + 1, j + 1] = F_n
                F_s_array[i + 1, j + 1] = F_s

    A_figo = A.toarray()
    b = b.toarray()
    return A, b, a_p_return, F_e_array, F_w_array, F_n_array, F_s_array


def solve_momentum(u, v, p, x_ap, y_ap, Re, nx, ny, direction, alpha_uv_solver, alpha_f ,assem=True):
    A, b, a_p_return, F_e_array, F_w_array, F_n_array, F_s_array = build_Ab(u=u, v=v, p=p, x_ap=x_ap, y_ap=y_ap, Re=Re,
                                                                            nx=nx, ny=ny, direction=direction, alpha_uv_solver=alpha_uv_solver,
                                                                            alpha_f=alpha_f, assem=assem)

    a_test = A.toarray()
    if assem == True:
        vel, err, time = solve_cgs(A, b, 1e-9)

        vel_array = vel.reshape(ny,nx)

        if direction == 'x':
            vel_return = np.copy(u)
            vel_return[1:nx + 1, 1:ny + 1] = vel_array
        elif direction == 'y':
            vel_return = np.copy(v)
            vel_return[1:nx + 1, 1:ny + 1] = vel_array
        else:
            raise ValueError('direction must be either x or y')
    else:
        vel_return = np.zeros((nx + 2, ny + 2))

    return vel_return, a_p_return, F_e_array, F_w_array, F_n_array, F_s_array


def update_F_ap(u, v, p, x_ap, y_ap, Re, nx, ny, alpha_uv_solver, alpha_f):

    _, x_ap, Fe_array, Fw_array, Fn_array, Fs_array = solve_momentum(u=u, v=v, p=p, x_ap=x_ap, y_ap=y_ap, Re=Re, nx=nx,
                                                                     ny=ny, direction='x', assem=False, alpha_uv_solver=alpha_uv_solver, alpha_f=alpha_f)
    _, y_ap, _, _, _, _ = solve_momentum(u=u, v=v, p=p, x_ap=x_ap, y_ap=y_ap, Re=Re, nx=nx,
                                                                     ny=ny, direction='y', assem=False, alpha_uv_solver=alpha_uv_solver, alpha_f=alpha_f)

    return x_ap, y_ap, Fe_array, Fw_array, Fn_array, Fs_array

def plot(vel):
    x_node = np.array([0.25, 0.5, 0.75, 1])
    y_node = np.array([1, 0.75, 0.5, 0.25])
    X, Y = np.meshgrid(x_node, y_node[::-1])
    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
    plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
    plt.pcolormesh(X, Y, vel, cmap='hot')
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.show()

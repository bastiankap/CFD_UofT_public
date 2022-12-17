import numpy as np

def gen_mesh(nx,ny):
    dx = 1/nx
    dy = 1/ny

    x_nodes = [dx/2]
    for i in np.arange(nx-1):
        x_nodes.append(x_nodes[-1]+dx)
    y_nodes = [1-dx/2]
    for i in np.arange(ny-1):
        y_nodes.append(y_nodes[-1]-dx)

    return dx,dy,x_nodes,y_nodes


def gen_ini(nx, ny):
    """
    generate initial guess for u,v,p
    :return: u,v,p as arrays including values for the boundaries
    NOTE: indeces for inner nodes go from 1 to n_rows-2!!!! so for np.arange it is (1,nx-1) since the last value is excluded
    """
    u_field = np.zeros((ny + 2, nx + 2))
    u_field[:, 0] = 0
    u_field[:, ny+1] = 0
    u_field[nx+1, :] = 0
    u_field[0, 1:nx+1] = 1

    v_field = np.zeros((ny + 2, nx + 2))
    v_field[:, 0] = 0
    v_field[:, ny+1] = 0
    v_field[0, :] = 0
    v_field[nx+1, :] = 0

    p_field = np.zeros((ny + 2, nx + 2))
    p_prime_field = np.zeros((ny+2, nx+2))

    x_momentum_ap = np.ones((ny + 2, nx + 2))
    x_momentum_ap[0,:] = 0
    x_momentum_ap[ny+1:] = 0
    x_momentum_ap[:,0] = 0
    x_momentum_ap[:,nx+1] = 0

    y_momentum_ap = np.ones((ny + 2, nx + 2))
    y_momentum_ap[0, :] = 0
    y_momentum_ap[ny + 1:] = 0
    y_momentum_ap[:, 0] = 0
    y_momentum_ap[:, nx + 1] = 0




    return u_field, v_field, p_field, p_prime_field, x_momentum_ap, y_momentum_ap


print('')

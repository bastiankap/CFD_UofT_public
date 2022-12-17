import numpy as np
from initial_conditions import *
from scipy.sparse import diags
from solver import solve_cgs
from momentum import update_F_ap

def gen_anb_p_prime(x_ap, y_ap, nx, ny, dx, dy, alpha_uv_solver):
    """
    solve pressure correction
    :param u: x velocity field
    :param v: y velocity field
    :param ap_array: ap values of all nodes in an array, used to interpolate ap at boundaries
    :return: pressure correction field
    """
    a_e = np.zeros((nx+2,ny+2))
    a_w = np.zeros((nx+2,ny+2))
    a_n = np.zeros((nx+2,ny+2))
    a_s = np.zeros((nx+2,ny+2))

    for i in np.arange(1,nx+1):
        for j in np.arange(1,ny+1):
            a_p_e = 1/2 * (x_ap[i, j] + x_ap[i, j + 1])
            a_p_w = 1/2 * (x_ap[i, j] + x_ap[i, j - 1])
            a_p_n = 1/2 * (y_ap[i, j] + y_ap[i - 1, j])
            a_p_s = 1/2 * (y_ap[i,j] + y_ap[i + 1, j])

            a_e[i,j] = alpha_uv_solver*dy/a_p_e
            a_w[i,j] = alpha_uv_solver*dy/a_p_w
            a_n[i,j] = alpha_uv_solver*dx/a_p_n
            a_s[i,j] = alpha_uv_solver*dx/a_p_s

    a_e[:,nx] = 0
    a_w[:,1] = 0
    a_n[1,:] = 0
    a_s[ny,:] = 0

    return a_e, a_w, a_n, a_s

def gen_source_p_prime(F_e, F_w, F_n, F_s, nx, ny, dx, dy):
    s_u = np.zeros((nx+2,ny+2))

    for i in np.arange(1,ny+1):
        for j in np.arange(1,nx+1):
            s_u[i,j] = (F_w[i,j] - F_e[i,j])*dy + (F_s[i,j]-F_n[i,j])*dx
    return s_u

def build_Ab_p_prime(a_e, a_w, a_s, a_n, s_u, nx, ny):
    a_p = np.zeros((ny, nx))
    a_e = a_e[1:ny+1,1:nx+1]
    a_w = a_w[1:ny+1,1:nx+1]
    a_n = a_n[1:ny+1,1:nx+1]
    a_s = a_s[1:ny+1,1:nx+1]
    s_u = s_u[1:ny+1, 1:nx+1]

    for i in np.arange(0,ny):
        for j in np.arange(0,nx):
            a_p[i,j] = a_e[i,j] + a_w[i,j] + a_n[i,j] + a_s[i,j]
    a_p = a_p.flatten()
    a_e = a_e.flatten()[:-1]
    a_w = a_w.flatten()[1:]
    a_s = a_s.flatten()[:-nx]
    a_n = a_n.flatten()[nx:]

    A = diags([a_p, -a_e, -a_w, -a_s, -a_n],[0,1,-1,nx,-nx]).toarray()

    b = s_u.flatten()
    return A,b

def solve_p_prime(u,v,p,Fe_array,Fw_array,Fn_array,Fs_array,x_ap,y_ap,Re,p_prime_old,nx,ny,dx, dy, alpha_uv_solver):
    a_e, a_w, a_n, a_s = gen_anb_p_prime(x_ap=x_ap, y_ap=y_ap, nx=nx, ny=ny, dx=dx, dy=dy, alpha_uv_solver=alpha_uv_solver)
    s_u = gen_source_p_prime(F_e=Fe_array, F_w=Fw_array, F_n=Fn_array, F_s=Fs_array, nx=nx, ny=ny, dx=dx, dy=dy)
    A, b = build_Ab_p_prime(a_e=a_e, a_w=a_w, a_s=a_s, a_n=a_n, s_u=s_u, nx=nx, ny=ny)

    p_prime, err, t = solve_cgs(A,b, tol=1e-9)
    p_prime_inner = p_prime.reshape(nx,ny)
    p_prime_ref = p_prime_inner[ny-1,0]
    p_prime_inner = p_prime_inner-p_prime_ref
    p_prime_complete = np.copy(p_prime_old)
    p_prime_complete[1:ny+1,1:nx+1] = p_prime_inner

    return p_prime_complete



def extrapolate_p_prime(p_prime, nx, ny):
    p_prime_ext = np.copy(p_prime)
    # east face
    j = 0
    for i in np.arange(1,ny+1):
        p_prime_ext[i,j] = 1.5 * p_prime_ext[i,j+1] - 0.5 * p_prime_ext[i,j+2]

    # west face
    j = nx + 1
    for i in np.arange(1,ny+1):
        p_prime_ext[i,j] = 1.5 * p_prime_ext[i, j-1] - 0.5 * p_prime_ext[i,j-2]

    # north face
    i = 0
    for j in np.arange(1,nx+1):
        p_prime_ext[i,j] = 1.5 * p_prime_ext[i+1, j] - 0.5 * p_prime_ext[i+2, j]

    # south face
    i = ny + 1
    for j in np.arange(1,nx+1):
        p_prime_ext[i,j] = 1.5 * p_prime_ext[i-1, j] - 0.5 * p_prime_ext[i-2, j]

    return p_prime_ext


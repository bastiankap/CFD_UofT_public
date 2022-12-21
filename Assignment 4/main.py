import numpy as np

from momentum import solve_momentum, gen_FD, update_F_ap
from initial_conditions import gen_ini, gen_mesh
import time
from pressure_correction import solve_p_prime
from plotting import plot_contour, plot_streamline
from velocity_correction import correct_velocity
from extrapolate_pressure import extrapolate
# initialize problem
t_start = time.time()
nx = 20
ny = nx
Re = 100
ext_case = 'zero gradient'
n_consecutive = 0

dx, dy, x_nodes, y_nodes = gen_mesh(nx, ny)
u, v, p, p_prime, x_ap, y_ap = gen_ini(nx, ny)
# define under-relaxation
alpha_p_prime = 0.3
alpha_u_prime = 1
alpha_v_prime = 1
alpha_momentum = 1
alpha_F = 0.7
alpha_uv_solver = 0.7
p_prime_max = 1
i = 0
i_max = 3000
p_prime_collected = np.array([])

while p_prime_max > 1e-4:

    print(f'iteration {i+1}')
    # solve x and y momentum
    u_star, x_ap, _, _, _, _ = solve_momentum(u=u, v=v, p=p, x_ap=x_ap, y_ap=y_ap, Re=Re, nx=nx,
                                                                     ny=ny, direction='x', alpha_uv_solver=alpha_uv_solver, assem=True, alpha_f=alpha_F)

    v_star, y_ap, _, _, _, _ = solve_momentum(u=u, v=v, p=p, x_ap=x_ap, y_ap=y_ap, Re=Re, nx=nx,
                                                                     ny=ny, direction='y', alpha_uv_solver=alpha_uv_solver, assem=True, alpha_f=alpha_F)

    _, _, Fe_array, Fw_array, Fn_array, Fs_array = update_F_ap(u=u_star, v=v_star, p=p, x_ap=x_ap, y_ap=y_ap, Re=Re, nx=nx,
                                                                     ny=ny, alpha_uv_solver=alpha_uv_solver, alpha_f=alpha_F)
    # solve pressure correction
    p_prime = solve_p_prime(u=u_star, v=v_star, p=p, Fe_array=Fe_array, Fw_array=Fw_array, Fn_array=Fn_array, Fs_array=Fs_array,
                            x_ap=x_ap, y_ap=y_ap, Re=Re, p_prime_old=p_prime, nx=nx, ny=ny, dx=dx, dy= dy, alpha_uv_solver=alpha_uv_solver)


    p_prime = extrapolate(array=p_prime, nx=nx, ny=ny, case=ext_case)
    # correct pressure
    p[1:ny + 1, 1:nx + 1] = p[1:ny + 1, 1:nx + 1] + alpha_p_prime * p_prime[1:ny + 1, 1:nx + 1]
    p = extrapolate(array=p, nx=nx, ny=ny, case = ext_case)
    x_ap = extrapolate(array=x_ap, nx=nx, ny=ny, case = ext_case)
    y_ap = extrapolate(array=y_ap, nx=nx, ny=ny, case = ext_case)

    # correct velocities
    u_cor, v_cor = correct_velocity(alpha_u_prime = alpha_u_prime, alpha_v_prime = alpha_v_prime, u =u_star, v=v_star, p_prime=p_prime, x_ap=x_ap, y_ap=y_ap, dx=dx, dy=dy, nx=nx, ny=ny)

    u[1:ny+1, 1:nx+1] = (1-alpha_momentum)*u_star[1:ny+1, 1:nx+1] + alpha_momentum * u_cor[1:ny+1, 1:nx+1]
    v[1:ny+1, 1:nx+1] = (1-alpha_momentum)*v_star[1:ny+1, 1:nx+1] + alpha_momentum * v_cor[1:ny+1, 1:nx+1]

    p_prime_max = np.max(np.abs(p_prime))
    vel_diff_max = max(np.max(np.abs(u_star - u)), np.max(np.abs(v_star - v)))

    if p_prime_max == 0:
        print('p_prime_max went to 0')
    i += 1

    p_prime_collected = np.append(p_prime_collected,p_prime_max)

    if i == i_max:
        print('Max. amount of iterations reached')
        print(f'Max. p_prime: {p_prime_max}')
        break
print('\n')
print(f'time: {(time.time()-t_start)/60} min')
print(f'number of iterations: {i}')
print(f'Max. p_prime: {p_prime_max}')
print(f'Max. velocity change: {vel_diff_max}')

plot_contour(x_nodes=x_nodes, y_nodes=y_nodes, array=p_prime[1:ny + 1, 1:nx + 1], title='p_prime')
plot_contour(x_nodes=x_nodes, y_nodes=y_nodes, array=p[1:ny + 1, 1:nx + 1], title='p')
plot_contour(x_nodes=x_nodes, y_nodes=y_nodes, array=u[1:ny + 1, 1:nx + 1], title='u')
plot_contour(x_nodes=x_nodes, y_nodes=y_nodes, array=v[1:ny + 1, 1:nx + 1], title='v')
plot_streamline(x_nodes=x_nodes,y_nodes=y_nodes,u_field=u[1:ny + 1, 1:nx + 1], v_field=v[1:ny + 1, 1:nx + 1], Re=Re, nx=nx)


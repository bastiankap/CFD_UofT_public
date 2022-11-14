import numpy as np
from scipy.sparse import dok_matrix
import time
from solver import solve_cgs
import matplotlib.pyplot as plt

t1 = time.time()

#Define parameters
rho = 20
h_f = 10
phi_x0 = 10
phi_xL = 100
phi_ext = 300
L_x = 1
L_y = 1
nx = 3
ny = 3
delta_x = L_x/nx
delta_y = L_y/ny

# a parameters for all inne nodes
a_w = rho*delta_y/delta_x
a_e = rho*delta_y/delta_x
a_s = rho*delta_x/delta_y
a_n = rho*delta_x/delta_y
a_p = a_n+a_e+a_s+a_w

# initialize matrix A,b
A = dok_matrix((nx*ny,nx*ny))
b = dok_matrix((nx*ny,1))

#populate values for inner nodes
#i:row of mesh [0:ny-1], j: column of mesh[0:nx-1]
for i in np.arange(1,ny-1): #NOTE: arange only goes until ny-2 so right boundary is also excluded!
    for j in np.arange(1,nx-1):
        k = (i*nx+j)
        A[k,k] = a_p
        A[k,k+1] = -a_e
        A[k,k-1] = -a_w
        A[k,k+nx] = -a_s
        A[k,k-nx] = -a_n

# populate values for left boundary
j = 0
s_p_left = -2*rho*delta_y/delta_x
s_u_left = 2*rho*delta_y/delta_x*phi_x0
a_w_left = 2*rho*delta_y/delta_x
a_p_left = a_e+a_n+a_s-s_p_left
for i in np.arange(1,ny-1):
    k = (i*nx+j)
    A[k,k] = a_p_left
    A[k,k+1] = -a_e
    A[k,k+nx] = -a_s
    A[k,k-nx] = -a_n
    b[k] = s_u_left

# populate values for right boundary
j=nx-1
s_p_right = -2*rho*delta_y/delta_x
s_u_right = 2*rho*delta_y/delta_x*phi_xL
a_p_right = a_s+a_w+a_n-s_p_right
for i in np.arange(1,ny-1):
    k = (i*nx+j)
    A[k,k] = a_p_right
    A[k,k-1] = -a_w
    A[k,k+nx] = -a_s
    A[k,k-nx] = -a_n
    b[k] = s_u_right

# populate values for lower boundary (a_s = 0)
i = ny-1
s_p_lower = 0
s_u_lower = 0
a_p_lower = a_n + a_w + a_e - s_p_lower
for j in np.arange(1,nx-1):
    k = (i * nx + j)
    A[k, k] = a_p_lower
    A[k, k + 1] = -a_e
    A[k, k - 1] = -a_w
    A[k, k - nx] = -a_n
    b[k] = s_u_lower

# populate values for upper boundary
i = 0
s_p_upper = -(rho/(h_f*delta_y/2 - rho))
s_u_upper = h_f*(phi_ext-(((delta_y/2)*(h_f-phi_ext))/(h_f*delta_y/2 - rho)))
a_p_upper = a_w+a_e+a_s-s_p_upper
for j in np.arange(1,nx-1):
    k = (i * nx + j)
    A[k, k] = a_p_upper
    A[k, k + 1] = -a_e
    A[k, k - 1] = -a_w
    A[k, k + nx] = -a_s
    b[k] = s_u_upper

# upper left corner
k = 0

s_p_ul = s_p_upper+s_p_left
s_u_ul = s_u_upper + s_u_left
a_p_ul = a_e+a_s-s_p_ul
A[k,k] = a_p_ul
A[k,k+1] = -a_e
A[k,k+nx] = -a_s
b[k] = s_u_ul

# upper right corner
k = nx-1
s_p_ur = s_p_upper + s_p_right
s_u_ur = s_u_upper + s_u_right
a_p_ur = a_s+a_w-s_p_ur
A[k,k] = a_p_ur
A[k,k-1] = -a_w
A[k,k+nx] = -a_s
b[k] = s_u_ur

#lower left corner
k = nx*(ny-1)
s_p_ll = s_p_left+s_p_lower
s_u_ll = s_u_lower+s_u_left
a_p_ll = a_e+a_n-s_p_ll
A[k, k] = a_p_ll
A[k, k + 1] = -a_e
A[k, k - nx] = -a_n
b[k] = s_u_ll

#lower right corner
k = (nx*ny)-1
s_p_lr = s_p_lower +s_p_right
s_u_lr = s_u_lower+s_u_right
a_p_lr = a_n+a_w-s_p_lr
A[k, k] = a_p_lr
A[k, k - 1] = -a_w
A[k, k - nx] = -a_n
b[k] = s_u_lr

b.toarray()
phi,err,t_solve = solve_cgs(A,b.toarray())
print(phi)
print(err)
print(t_solve)
print(t1-time.time())

A_mat = A.toarray()
b_vec = b.toarray()

#visualize
M_show_list = []

for i in np.arange(0,ny):
   M_show_list.append(phi[i*ny:(i*ny+nx)])

M_show = np.asarray(M_show_list)
plt.imshow(M_show, cmap='hot', interpolation='nearest')
plt.show()

print(f'max phi= {max(phi)}')
print(f'min phi= {min(phi)}')

print(f'total time: {time.time()-t1}')
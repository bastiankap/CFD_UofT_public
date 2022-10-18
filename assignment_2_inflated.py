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
nx = 1000
ny = 1000

#generate mesh
r = 1.000000001 #inflation factor
x_0 = ((1-r)/(1-r**(nx/2)))*L_x/2 #boundary mesh
y_0 = ((1-r)/(1-r**(ny/2)))*L_y/2 #boundary mesh
width_x = [x_0]
width_y = [y_0]

while round(sum(width_x), ndigits=10) < L_x/2:
    x = width_x[-1] * r
    width_x.append(x)
width_x = width_x + list(reversed(width_x))

while round(sum(width_y), ndigits=10) < L_y/2:
    y = width_y[-1] * r
    width_y.append(y)
width_y = width_y + list(reversed(width_y))

delta_x = [x_0]
delta_y = [y_0]

for i in np.arange(0,len(width_x)-1):
    delta_x.append(width_x[i]/2 + width_x[i+1]/2)
delta_x.append(x_0)
for i in np.arange(0, len(width_y)-1):
    delta_y.append(width_y[i] / 2 + width_y[i + 1] / 2)
delta_y.append(y_0)

#generate a values
a_w = np.zeros(shape=(ny,nx))
a_e = np.zeros(shape=(ny,nx))
a_n = np.zeros(shape=(ny,nx))
a_s = np.zeros(shape=(ny,nx))
a_p = np.zeros(shape=(ny,nx))
#generate values for a_s,a_n,a_w,a_e
for i in np.arange(0,ny-1):
    for j in np.arange(0,nx):
        a_s[i,j] = rho * width_x[j] / delta_y[i+1]

for i in np.arange(1,ny):
    for j in np.arange(0,nx):
        a_n[i, j] = rho * width_x[j] / delta_y[i]

for i in np.arange(0,ny):
    for j in np.arange(1,nx):
        a_w[i,j] = rho * width_y[i] / delta_x[j]

for i in np.arange(0, ny):
    for j in np.arange(0, nx - 1):
        a_e[i,j] = rho * width_y[i] / delta_x[j+1]

# generate source terms for the boundaries
s_u = np.zeros(shape=(ny,nx))
s_p = np.zeros(shape=(ny,nx))

# source terms on upper boundary (i = 0)
i = 0
for j in np.arange(0,nx):
    s_u[i,j] = s_u[i,j] + width_x[j]*(h_f*(phi_ext-(((delta_y[i]/2)*(h_f-phi_ext))/(h_f*delta_y[i]/2 - rho))))
    s_p[i,j] = s_p[i,j] - width_x[j]*(h_f*rho/(h_f*delta_y[i]/2 - rho))
# source terms for lower boundary
i = ny-1
for j in np.arange(0,nx):
    s_u[i,j]= s_u[i,j] + 0
    s_p[i,j]= s_p[i,j] + 0

# source terms for left boundary
j = 0
for i in np.arange(0,ny):
    s_u[i,j]= s_u[i,j] + 2 * rho * width_y[i] / delta_x[j] * phi_x0
    s_p[i,j]= s_p[i,j] + -2 * rho * width_y[i] / delta_x[j]

# source terms for right boundary
j = nx-1
for i in np.arange(0,ny):
    s_u[i,j]= s_u[i,j] + 2 * rho * width_y[i] / delta_x[j+1] * phi_xL
    s_p[i,j]= s_p[i,j] + -2 * rho * width_y[i] / delta_x[j+1]

# generate a_p
for i in np.arange(0,ny):
    for j in np.arange(0,nx):
        a_p[i,j] = a_e[i,j] + a_w[i,j] + a_s[i,j] + a_n[i,j] - s_p[i,j]

# build matrix A and vector b
A = dok_matrix((nx * ny, nx * ny))
b = dok_matrix((nx * ny, 1))

for i in np.arange(0,ny):
    for j in np.arange(0,nx):
        a_p = a_n[i,j]+a_s[i,j]+a_w[i,j]+a_e[i,j]-s_p[i,j]

        k = (i * nx + j)

        A[k, k] = a_p
        if k+1 <= (nx*ny)-1:
            A[k, k + 1] = -a_e[i,j]
        if k-1 >= 0:
            A[k, k - 1] = -a_w[i,j]
        if k+nx <= (nx*ny)-1:
            A[k, k + nx] = -a_s[i,j]
        if k-nx >= 0:
            A[k, k - nx] = -a_n[i,j]
        b[k] = s_u[i,j]

# A_mat = A.toarray()
# b_vec = b.toarray()

phi,err,t_solve = solve_cgs(A,b.toarray())
print(phi)

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
from test_matrices import *
from solver import *

n = 1000

A_band, A_textbook, A_rand, A_sym, b_rand, b_ones, b_textbook = generate_test_matrices(n)
A_csr, b_csr = generate_banded_csr(10000000)

matrices = {"banded A": A_band,  "random A": A_rand}#, "symmetric A": A_sym,}
vectors = {"random b": b_rand}

for key_A in matrices.keys():
    for key_b in vectors.keys():
        print(f"A matrix: {key_A}")
        print(f"b vector: {key_b}")

        x_cg, error_cg, total_time_cg = solve_cg(matrices[key_A], vectors[key_b],precon=True, maxiter=10000, tol=1e-10)
        print(f"Error cg: {np.max(error_cg)}, Total time cg: {total_time_cg}")

        x_cgs, error_cgs, total_time_cgs = solve_cgs(matrices[key_A], vectors[key_b], precon=True, maxiter=10000, tol=1e-10)
        print(f"Error cgs: {np.max(error_cgs)}, Total time cgs: {total_time_cgs}")

        x_bicgstab, error_bicgstab, total_time_bicgstab = solve_bicgstab(matrices[key_A], vectors[key_b], precon=True, maxiter=10000, tol=1e-10)
        print(f"Error bicgstab: {np.max(error_bicgstab)}, Total time bicgstab: {total_time_bicgstab}")

        x_lgmres, error_lgmres, total_time_lgmres = solve_lgmres(matrices[key_A], vectors[key_b], precon=True, maxiter=10000, tol=1e-10)
        print(f"Error lgmres: {np.max(error_lgmres)}, Total time lgmres: {total_time_lgmres}")

        x_gcrotmk, error_gcrotmk, total_time_gcrotmk = solve_gcrotmk(matrices[key_A], vectors[key_b], precon=True, maxiter=10000, tol=1e-10)
        print(f"Error gcrotmk: {np.max(error_gcrotmk)}, Total time gcrotmk: {total_time_gcrotmk}")

        print(":::::::::::::::::::::::::::::::::::::::::::::::::")

print("A_banded as csr, b_rand as csr")

x_cg, error_cg, total_time_cg = solve_cg(A_csr, b_csr, precon=True, maxiter=10000, tol=1e-10)
print(f"Error cg: {np.max(error_cg)}, Total time cg: {total_time_cg}")

x_cgs, error_cgs, total_time_cgs = solve_cgs(A_csr, b_csr, precon=True, maxiter=10000, tol=1e-10)
print(f"Error cgs: {np.max(error_cgs)}, Total time cgs: {total_time_cgs}")

x_bicgstab, error_bicgstab, total_time_bicgstab = solve_bicgstab(A_csr, b_csr, precon=True,
                                                                 maxiter=10000, tol=1e-10)
print(f"Error bicgstab: {np.max(error_bicgstab)}, Total time bicgstab: {total_time_bicgstab}")

x_lgmres, error_lgmres, total_time_lgmres = solve_lgmres(A_csr, b_csr, precon=True, maxiter=10000, tol=1e-10)
print(f"Error lgmres: {np.max(error_lgmres)}, Total time lgmres: {total_time_lgmres}")

x_gcrotmk, error_gcrotmk, total_time_gcrotmk = solve_gcrotmk(A_csr, b_csr, precon=True,
                                                             maxiter=10000, tol=1e-10)
print(f"Error gcrotmk: {np.max(error_gcrotmk)}, Total time gcrotmk: {total_time_gcrotmk}")

print("::::::::::::::::::::::::::::::::::::::::::::::::::")
print("Textbook example")
x_textbook, error_textbook, total_time_textbook = solve_gcrotmk(A_textbook, b_textbook, precon=True,
                                                             maxiter=10000, tol=1e-10)
print(f"Error gcrotmk: {np.max(error_gcrotmk)}, Total time gcrotmk: {total_time_gcrotmk}")
print(x_textbook)



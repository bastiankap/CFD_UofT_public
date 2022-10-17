from assignment1_solver import *
from test_matrices import *

n = 1000

A_band, A_textbook, A_rand, A_sym, b_rand, b_ones, b_textbook = generate_test_matrices(n)
A_csr, b_csr = generate_banded_csr(10000000)

matrices = {"banded A": A_band,  "random A": A_rand, "symmetric A": A_sym,}
vectors = {"random b": b_rand}

for key_A in matrices.keys():
    for key_b in vectors.keys():
        print(f"A matrix: {key_A}")
        print(f"b vector: {key_b}")

        x_cgs, error_cgs, total_time_cgs = solve_cgs(matrices[key_A], vectors[key_b], precon=True, maxiter=10000, tol=1e-10)
        print(f"Error cgs: {np.max(error_cgs)}, Total time cgs: {total_time_cgs}")

        print(":::::::::::::::::::::::::::::::::::::::::::::::::")

x_cgs, error_cgs, total_time_cgs = solve_cgs(A_csr, b_csr, precon=True, maxiter=10000, tol=1e-10)
print(f"Error cgs: {np.max(error_cgs)}, Total time cgs: {total_time_cgs}")

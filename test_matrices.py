from scipy.sparse import diags
import numpy as np
from scipy.sparse import rand


def generate_banded_csr(n):
    A_band = diags([3, -1, 2, -1, 3], [-3, -1, 0, 1, 3], shape=(n, n), format='csr')
    b = rand(n,1, density=1).toarray()
    results = [A_band, b]

    return results

def generate_test_matrices(n):
    A_band = diags([3, -1, 2, -1, 3], [-3,-1, 0, 1,3], shape=(n, n)).toarray()
    A_textbook = np.diag([20, 15, 15, 15, 10]) + \
         np.diag([-5, -5, -5, -5], k=1) + \
         np.diag([-5, -5, -5, -5], k=-1)  # matrix out of text book
    A_rand = rand(n,n, density= 0.1).toarray() + np.eye(n,n)
    A_sym_tool =  np.random.random_integers(-2000,2000,size=(n,n))
    A_sym = (A_sym_tool + A_sym_tool.T)/2

    b_rand = np.random.rand(n, 1)
    b_ones = np.ones((n, 1))
    b_textbook = [1100, 100, 100, 100, 100]

    results = (A_band, A_textbook, A_rand, A_sym, b_rand, b_ones, b_textbook)

    return results

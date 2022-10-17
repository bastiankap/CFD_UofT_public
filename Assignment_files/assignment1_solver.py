import scipy.sparse.linalg as lng
from scipy.sparse import csr_matrix, csc_matrix
import time
import numpy as np

def solve_cgs(A, b, tol=1e-05, maxiter=1000, precon=True):
    """
    This function solves systems of form Ax = b using 'Conjugate Gradient Squared iteration'
    :param A: A matrix, can be np.array or csr matrix
    :param b: b vector
    :param tol: termination criterion: max. approximation error. Default: 1e-05
    :param maxiter: termination criterion: max. number of iterations Default: 1000
    :param precon: if True, algorithm uses an approximation of the inverse of A to help convergence.
                   Preconditioning can be turned off, in case it leads to problems in the future. Default:True
    :return: x: result of solving the system
             error: |error| = |Ax-b|
             total_time: total calculation time
    """
    t0 = time.time() # start timer
    A_csr = csr_matrix(A) # convert A into a csr matrix. no problem if A already is a csr matrix

    if precon == True:  # using gcrotmk with preconditioner

        # approximating inverse of A to use as preconditioner
        A_csc = csc_matrix(A)
        try:  # try to use splu since it is more robust. might raise memory error on dense matrices
            A_splu = lng.splu(A_csc)
        except: # use spilu if splu raises memory error
            A_splu = lng.spilu(A_csc)
            print("spilu was used due to an error in splu")
        finally:
            M = lng.LinearOperator(np.shape(A), A_splu.solve) # precontitioner

        x, exitcode = lng.cgs(A_csr, b, maxiter=maxiter, tol=tol, M=M)  # solve
    else:
        x, exitcode = lng.cgs(A_csr, b, maxiter=maxiter, tol=tol) # solve without preconditioning

    error = np.absolute(A_csr.dot(x) - np.transpose(b))  # calculate the accuracy # calculate residual error

    if exitcode == 0: # exitcode of scipy.sparse.linalge.cgs gives information about convergence (see docomentation)
        print("convergence achieved")
    elif exitcode > 0:
        print("desired tolerance could not be achieved")
    else:
        raise Exception('Exitcode < 0')

    t1 = time.time() # end timer
    total_time = t1 - t0
    return x, error, total_time
import scipy.sparse.linalg as lng
from scipy.sparse import csr_matrix, csc_matrix
import time
import numpy as np


def solve_direct(A, b, tol=1e-05, maxiter=1000):
    # This function receives a sparse nxn matrix and a nx1 column matrix to solve for Ax = b using Conjugate Gradient
    t0 = time.time()
    A_csr = csr_matrix(A)

    x = lng.spsolve(A_csr, b)  # BICGstab algorithm to solve equations

    error = np.absolute(A_csr.dot(x) - np.transpose(b))  # calculate the accuracy
    t1 = time.time()
    total_time = t1 - t0
    return x, error, total_time


def solve_cg(A, b, tol=1e-05, maxiter=1000, precon=True):
    # This function receives a sparse nxn matrix and a nx1 column matrix to solve for Ax = b using Conjugate Gradient
    t0 = time.time()
    A_csr = csr_matrix(A)

    if precon == True:  # possibility to deactive the preconditioner in case it leads to problems

        A_csc = csc_matrix(A)
        try:  # try to use splu since it is more robust. might raise memory error on dense matrices
            A_splu = lng.splu(A_csc)
        except:
            A_splu = lng.spilu(A_csc)
            print("spilu was used due to an error in splu")
        finally:
            M = lng.LinearOperator(np.shape(A), A_splu.solve)

        x, exitcode = lng.cg(A_csr, b, maxiter=maxiter, tol=tol, M=M)  # BICGstab algorithm to solve equations
    else:
        x, exitcode = lng.cg(A_csr, b, maxiter=maxiter, tol=tol)

    error = np.absolute(A_csr.dot(x) - np.transpose(b))  # calculate the accuracy

    if exitcode == 0:
        print("convergence achieved")
    elif exitcode > 0:
        print("desired tolerance could not be achieved")
    else:
        raise Exception('Exitcode < 0')

    t1 = time.time()
    total_time = t1 - t0
    return x, error, total_time


def solve_cgs(A, b, tol=1e-05, maxiter=1000, precon=True):
    # This function receives a sparse nxn matrix and a nx1 column matrix to solve for Ax = b using Conjugate Gradient
    t0 = time.time()
    A_csr = csr_matrix(A)

    if precon == True:  # possibility to deactive the preconditioner in case it leads to problems

        A_csc = csc_matrix(A)
        try:  # try to use splu since it is more robust. might raise memory error on dense matrices
            A_splu = lng.splu(A_csc)
        except:
            A_splu = lng.spilu(A_csc)
            print("spilu was used due to an error in splu")
        finally:
            M = lng.LinearOperator(np.shape(A), A_splu.solve)

        x, exitcode = lng.cgs(A_csr, b, maxiter=maxiter, tol=tol, M=M)  # BICGstab algorithm to solve equations
    else:
        x, exitcode = lng.cgs(A_csr, b, maxiter=maxiter, tol=tol)

    error = np.absolute(A_csr.dot(x) - np.transpose(b))  # calculate the accuracy

    if exitcode == 0:
        print("convergence achieved")
    elif exitcode > 0:
        print("desired tolerance could not be achieved")
    else:
        raise Exception('Exitcode < 0')

    t1 = time.time()
    total_time = t1 - t0
    return x, error, total_time


def solve_bicgstab(A, b, tol=1e-05, maxiter=1000, precon=True):
    # This function receives a sparse nxn matrix and a nx1 column matrix to solve for Ax = b using BIConjugate Gradient
    t0 = time.time()

    A_csr = csr_matrix(A)  # If A is already a csr that is no problem

    if precon == True:  # possibility to deactive the preconditioner in case it leads to problems

        A_csc = csc_matrix(A)
        try:  # try to use splu since it is more robust. might raise memory error on dense matrices
            A_splu = lng.splu(A_csc)
        except:
            A_splu = lng.spilu(A_csc)
            print("spilu was used due to an error in splu")
        finally:
            M = lng.LinearOperator(np.shape(A), A_splu.solve)

        x, exitcode = lng.bicgstab(A_csr, b, maxiter=maxiter, tol=tol, M=M)  # BICGstab algorithm to solve equations
    else:
        x, exitcode = lng.bicgstab(A_csr, b, maxiter=maxiter, tol=tol)

    error = np.absolute(A_csr.dot(x) - np.transpose(b))  # calculate the accuracy

    if exitcode == 0:
        print("convergence achieved")
    elif exitcode > 0:
        print("desired tolerance could not be achieved")
    else:
        raise Exception('Exitcode < 0')

    t1 = time.time()
    total_time = t1 - t0
    return x, error, total_time


def solve_lgmres(A, b, tol=1e-05, maxiter=1000, precon=True):
    # This function receives a sparse nxn matrix and a nx1 column matrix to solve for Ax = b using Conjugate Gradient
    t0 = time.time()
    A_csr = csr_matrix(A)

    if precon == True:  # possibility to deactive the preconditioner in case it leads to problems

        A_csc = csc_matrix(A)
        try:  # try to use splu since it is more robust. might raise memory error on dense matrices
            A_splu = lng.splu(A_csc)
        except:
            A_splu = lng.spilu(A_csc)
            print("spilu was used due to an error in splu")
        finally:
            M = lng.LinearOperator(np.shape(A), A_splu.solve)

        x, exitcode = lng.lgmres(A_csr, b, maxiter=maxiter, tol=tol, M=M)  # BICGstab algorithm to solve equations
    else:
        x, exitcode = lng.lgmres(A_csr, b, maxiter=maxiter, tol=tol)

    error = np.absolute(A_csr.dot(x) - np.transpose(b))  # calculate the accuracy

    if exitcode == 0:
        print("convergence achieved")
    elif exitcode > 0:
        print("desired tolerance could not be achieved")
    else:
        raise Exception('Exitcode < 0')

    t1 = time.time()
    total_time = t1 - t0
    return x, error, total_time


def solve_gcrotmk(A, b, tol=1e-05, maxiter=1000, precon=True):
    """
    This function solves systems of form Ax = b using 'Generalized conjugate residual with inner orthogonalization
    and outer truncation'
    :param A: A matrix, can be np.array or csr matrix
    :param b: b vector
    :param tol: termination criterion: max. approximation error. Default: 1e-05
    :param maxiter: termination criterion: max. number of iterations Default: 1000
    :param precon: if True, algorithm uses an approximation of the inverse of A to help conversion.
                   Preconditioning can be turned off, in case it leads to problems in the future. Default:True
    :return: x: result of solving the system
             error: |error| = |Ax-b|
             total_time: total calculation time
    """

    t0 = time.time()
    A_csr = csr_matrix(A) # convert A into csr matrix

    if precon == True:  # using gcrotmk with preconditioner

        A_csc = csc_matrix(A)
        try: # try to use splu since it is more robust. might raise memory error on dense matrices
            A_splu = lng.splu(A_csc)
        except:
            A_splu = lng.spilu(A_csc) # use spilu instead of splu (for large, dense matrices)
            print("spilu was used due to an error in splu")
        finally:
            M = lng.LinearOperator(np.shape(A), A_splu.solve)

        x, exitcode = lng.gcrotmk(A_csr, b, maxiter=maxiter, tol=tol, M=M)
    else:  # using gcrotmk without preconditioner
        x, exitcode = lng.gcrotmk(A_csr, b, maxiter=maxiter, tol=tol)

    error = np.absolute(A_csr.dot(x) - np.transpose(b))  # residual error

    if exitcode == 0:  # exitcode of lng.gcrotmk gives information about convergence (see docomentation)
        print("convergence achieved")
    elif exitcode > 0:
        print("desired tolerance could not be achieved")
    else:
        raise Exception('Exitcode < 0')

    t1 = time.time()
    total_time = t1 - t0
    return x, error, total_time

import numpy as np


def correct_velocity(alpha_u_prime, alpha_v_prime, u, v, p_prime, x_ap, y_ap, dx, dy, nx, ny):
    u_correct = np.copy(u)
    v_correct = np.copy(v)

    for i in np.arange(1, ny + 1):
        for j in np.arange(1, nx + 1):
            p_prime_P = p_prime[i, j]
            p_prime_E = p_prime[i, j + 1]
            p_prime_W = p_prime[i, j - 1]
            p_prime_N = p_prime[i - 1, j]
            p_prime_S = p_prime[i + 1, j]
            u_correct[i, j] = u_correct[i, j] + alpha_u_prime / x_ap[i, j] * (
                        (p_prime_W + p_prime_P) / 2 - (p_prime_E + p_prime_P) / 2)
            v_correct[i, j] = v_correct[i, j] + alpha_v_prime / y_ap[i, j] * (
                        (p_prime_P + p_prime_S) / 2 - (p_prime_N + p_prime_P) / 2)

    return u_correct, v_correct

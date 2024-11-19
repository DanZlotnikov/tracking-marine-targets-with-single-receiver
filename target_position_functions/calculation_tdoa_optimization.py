import numpy as np
from numpy import sum, tile, round
from numpy.linalg import norm
from utils.time_utils import calculate_time_difference


def dis(p_a, p_b):
    print(p_a.shape)
    return ((p_a[:, 0] - p_b[:, 0]) ** 2 +
            (p_a[:, 1] - p_b[:, 1]) ** 2 +
            (p_a[:, 2] - p_b[:, 2]) ** 2) ** 0.5


def hyperbola(p_t, p_a, p_b, range_dif):
    return (dis(p_t, p_a) - dis(p_t, p_b) - range_dif) ** 1


def objective_function(p_t, p_a, p_b, range_dif, mu_lst):
    return sum(mu_lst * hyperbola(p_t, p_a, p_b, range_dif) ** 2, 0)


def gradient(p_t, p_a, p_b, range_dif, mu_lst, symbol):
    idx = ord(symbol) - ord('x')
    grad = 2 * mu_lst * (
            (p_t[:, idx] - p_a[:, idx]) / dis(p_t, p_a) -
            (p_t[:, idx] - p_b[:, idx]) / dis(p_t, p_b)) * (
                   dis(p_t, p_a) - dis(p_t, p_b) - range_dif)
    return grad


def objective_gradient(p_t, p_a, p_b, range_dif, mu_lst, is_2d):
    grad = np.zeros(3)

    grad_x = gradient(p_t, p_a, p_b, range_dif, mu_lst, 'x')
    grad_y = gradient(p_t, p_a, p_b, range_dif, mu_lst, 'y')

    if is_2d:
        grad[0:2] = sum([grad_x, grad_y], 1)
    else:
        grad_z = gradient(p_t, p_a, p_b, range_dif, mu_lst, 'z')
        grad = sum([grad_x, grad_y, grad_z], 1)

    return grad


def tdoa_optimization(rov_lst, t_lst, velocity, p_t, is_2d):
    return steepest_descent(rov_lst, t_lst, velocity, p_t, is_2d)


def steepest_descent(rov_lst, t_lst, velocity, p_t, is_2d):
    maxIter = 100
    eps = 0.01
    rov_lst = np.array(rov_lst)
    n_points = rov_lst.shape[0]

    mu_lst = np.flip(np.array([0.5 ** (i + 1) for i in range(len(rov_lst) - 1)]))
    time_dif = calculate_time_difference(t_lst)
    range_dif = time_dif * velocity
    # range_dif = (t_lst[1:n_points] - t_lst[0:n_points - 1]) * velocity

    p_b = rov_lst[0:n_points - 1]
    p_a = rov_lst[1:n_points]

    p_t_lst = tile(p_t, (n_points - 1, 1))
    objective_func_val = objective_function(p_t_lst, p_a, p_b, range_dif, mu_lst)
    error_val = norm(objective_func_val - 0)

    for idx in range(maxIter):

        if error_val < eps:
            break

        d = objective_gradient(p_t_lst, p_a, p_b, range_dif, mu_lst, is_2d)
        grad_F = objective_gradient(p_t_lst, p_a, p_b, range_dif, mu_lst, is_2d)
        alpha, is_failed = armijo_search(rov_lst, p_t, mu_lst, range_dif, grad_F, d)

        if is_failed:
            return np.array([]), True

        p_t = p_t - alpha * d

        p_t_lst = tile(p_t, (n_points - 1, 1))
        objective_func_prev_val = objective_func_val
        objective_func_val = objective_function(p_t_lst, p_a, p_b, range_dif, mu_lst)
        error_val = norm(objective_func_val - objective_func_prev_val)

    return np.array([p_t]), False


def armijo_search(rov_lst, p_t, mu_lst, range_dif, grad_f, d):
    alpha = 1
    betta = 0.5
    c = 1 / 1000

    n_points = rov_lst.shape[0]
    p_b = rov_lst[0:n_points - 1]
    p_a = rov_lst[1:n_points]
    p_t_lst = tile(p_t, (n_points - 1, 1))

    maxIter = 50

    for i in range(maxIter):
        p_t_new = p_t - alpha * d
        p_t_new_lst = tile(p_t_new, (n_points - 1, 1))

        objective_function_val = objective_function(p_t_lst, p_a, p_b, range_dif, mu_lst)
        objective_function_new_val = objective_function(p_t_new_lst, p_a, p_b, range_dif, mu_lst) + (
                c * alpha * (grad_f @ (- d)))

        if round(objective_function_new_val, 3) <= round(objective_function_val, 3):
            return alpha, False
        else:
            alpha = alpha * betta

    return alpha, True

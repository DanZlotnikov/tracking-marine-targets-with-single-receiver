import os
import shutil
import pickle
import numpy as np
from numpy import round
from numpy.linalg import norm

import matplotlib
font = {'family': 'normal',
        'weight': 'bold',
        'size': 7}

matplotlib.use("TkAgg")
matplotlib.rc('font', **font)
import matplotlib.pyplot as plt


# We will mark the following parameters:
# p_{i}, p_{i-1}, p_{i-2} = p_i, p_i_1, p_i_2

def dis(p_a, p_b):
    return ((p_a[0] - p_b[0]) ** 2 + (p_a[1] - p_b[1]) ** 2) ** 0.5


# Equal distances (radius) between Rover last locations and the target
def dif_dis_from_target(p_i, p_i_1, p_t):
    return dis(p_i, p_t) - dis(p_i_1, p_t)


# Equal distances between Rover next location to two last rover locations
def dif_dis_from_prev_locations(p_i, p_i_1, p_i_2):
    return dis(p_i, p_i_2) - dis(p_i, p_i_1)


def objective_function(p_i, p_i_1, p_i_2, p_t, mu_lst):
    return mu_lst[0] * (dif_dis_from_target(p_i, p_i_1, p_t)) ** 2 + \
           mu_lst[1] * (dif_dis_from_prev_locations(p_i, p_i_1, p_i_2) ** 2)


def grad_dif_dis_from_target(p_i, p_i_1, p_t, mu):
    return 2 * mu * dif_dis_from_target(p_i, p_i_1, p_t) * (p_i - p_t) / dis(p_i, p_t)


def grad_dif_dis_from_prev_locations(p_i, p_i_1, p_i_2, mu):
    return 2 * mu * dif_dis_from_prev_locations(p_i, p_i_1, p_i_2) * (
            (p_i - p_i_2) / dis(p_i, p_i_2) - (p_i - p_i_1) / dis(p_i, p_i_1))


def objective_gradient(p_i, p_i_1, p_i_2, p_t, mu_lst):
    return grad_dif_dis_from_target(p_i, p_i_1, p_t, mu_lst[0]) + \
           grad_dif_dis_from_prev_locations(p_i, p_i_1, p_i_2, mu_lst[1])


# Given the last two Rover locations, we calculate the next best location. This function can also
# calculate the next additional steps for the rover assuming the target is not moving.
def next_location_triangle(rov_lst, fish_loc, max_distance, n_steps=1, return_2D = True):
    point_lst = np.concatenate((rov_lst, np.array([fish_loc])))
    point_lst = np.copy(point_lst)
    n = len(point_lst) - 1
    next_location_lst = []
    for i in range(n_steps):
        res, is_failed = steepest_descent(point_lst, max_distance)
        point_lst[0:n - 1] = point_lst[1:n]
        point_lst[n - 1] = res
        if not return_2D:
            res = np.array([res[0], res[1], 0])
        next_location_lst.append(res)

    return np.array(next_location_lst)


# Given an initial guess, we will look for the best point that will reduce the cost function
def steepest_descent(point_lst, max_distance):
    maxIter = 100
    eps = 0.01
    is_failed = False

    mu_lst = np.array([0.17, 0.83])

    p_i_2, p_i_1 = point_lst[0:2]
    p_t = point_lst[2]
    p_i = point_lst[2] + np.array([0.1, -0.2])

    objective_func_val = objective_function(p_i, p_i_1, p_i_2, p_t, mu_lst)
    error_val = norm(objective_func_val - 0)

    for idx in range(maxIter):

        if error_val < eps:
            break

        d = objective_gradient(p_i, p_i_1, p_i_2, p_t, mu_lst)
        grad_F = objective_gradient(p_i, p_i_1, p_i_2, p_t, mu_lst)
        alpha, is_failed = armijo_search(p_i, p_i_1, p_i_2, p_t, mu_lst, grad_F, d, max_distance)

        p_i = p_i - alpha * d
        p_i = point_project(p_i, p_i_1, max_distance)

        objective_func_prev_val = objective_func_val
        objective_func_val = objective_function(p_i, p_i_1, p_i_2, p_t, mu_lst)
        error_val = norm(objective_func_val - objective_func_prev_val)

    return p_i, is_failed


# This function calculates the step size for steepest descent
def armijo_search(p_i, p_i_1, p_i_2, p_t, mu_lst, grad_f, d, max_distance):
    alpha = 1
    betta = 0.5
    c = 1 / 1000
    maxIter = 50

    for i in range(maxIter):
        p_i_new = p_i - alpha * d
        p_i_new = point_project(p_i_new, p_i_1, max_distance)

        objective_function_val = objective_function(p_i, p_i_1, p_i_2, p_t, mu_lst)
        objective_function_new_val = objective_function(p_i_new, p_i_1, p_i_2, p_t, mu_lst) + (
                c * alpha * np.matmul(grad_f, (- d)))

        if round(objective_function_new_val, 3) <= round(objective_function_val, 3):
            return alpha, False
        else:
            alpha = alpha * betta

    return alpha, True


# In case we get that the best point is not within reasonable range, we will take a closer point
def point_project(p_next, p_prev, max_distance):
    direction = (p_next - p_prev) / dis(p_next, p_prev)
    distance = dis(p_next, p_prev)

    if distance > max_distance:
        return p_prev + direction * max_distance
    elif distance < max_distance * 0.7:
        return p_prev + direction * (0.7 * max_distance)
    else:
        return p_next

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

if __name__ == '__main__':
    num_file = 83
    n_steps = 10
    path = f'{parent_dir}/plots/tests/test_' + str(num_file)

    if os.path.isdir(path + "/next_location_test"):
        shutil.rmtree(path + "/next_location_test")
    os.mkdir(path + "/next_location_test")

    with open(path + '/data_output' + str(num_file) + '.pkl', 'rb') as run_data_output:
        rover_lst, t_lst, fish_lst, idx_in_use_lst = pickle.load(run_data_output)

    n = len(t_lst)
    rover_lst = np.array(rover_lst)[:, :2]
    t_lst = np.array(t_lst)
    fish_lst = np.array(fish_lst)[:, :2]
    idx_in_use_lst = np.array(idx_in_use_lst)

    fig, ax = plt.subplots(1, 1, figsize=(16.0, 10.0))

    n_polygon = 3
    for i in np.arange(start=n_polygon, stop=n, step=1):
        last_fish_loc = fish_lst[i - n_polygon]
        next_point = next_location_triangle(rov_lst=rover_lst[i - 2:i],
                                            fish_loc=np.copy(last_fish_loc),
                                            max_distance=20,
                                            n_steps=n_steps,
                                            return_2D=False)

        ax.set_xlim([-40, 40])
        ax.set_ylim([-40, 40])
        ax.plot(rover_lst[i - n_polygon:i, 0], rover_lst[i - n_polygon:i, 1], c='red')
        ax.scatter(last_fish_loc[0], last_fish_loc[1])
        ax.plot([rover_lst[i - 1, 0], rover_lst[i, 0]], [rover_lst[i - 1, 1], rover_lst[i, 1]], c='pink')
        ax.plot([rover_lst[i - 1, 0], next_point[0, 0]], [rover_lst[i - 1, 1], next_point[0, 1]], c='green')
        if n_steps > 1:
            ax.plot(next_point[:n_steps, 0], next_point[:n_steps, 1], c='green')

        fig.show()
        plt.savefig(path + '/next_location_test/img' + str(1000 + i) + '.png', dpi=150)
        ax.cla()
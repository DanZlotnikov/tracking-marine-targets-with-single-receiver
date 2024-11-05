import os
import shutil
import pickle

import numpy as np
from numpy import arccos
from numpy.linalg import norm
from numpy.random import randn

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
    return ((p_a[:, 0] - p_b[:, 0]) ** 2 + (p_a[:, 1] - p_b[:, 1]) ** 2 + (p_a[:, 2] - p_b[:, 2]) ** 2) ** 0.5


# Equal distances (radius) between Rover last locations and the target
def equal_radius(p_i, p_i_1, p_t):
    return dis(p_i, np.array([p_t])) - norm(p_i_1 - p_t)


# Fixed distance from last location to the next Rover location
def fixed_distance(p_i, p_i_1, side_length):
    return dis(p_i, np.array([p_i_1])) - side_length


# Fixed distance (radius) from the rover to the target
def fixed_radius(p_i, p_t, radius_length):
    return dis(p_i, np.array([p_t])) - radius_length


# Fixed angle between the last three Rover locations
def fixed_angle(p_i, p_i_1, p_i_2, theta):
    dis_ab = norm(p_i_2 - p_i_1) ** 2
    dis_bc = dis(np.array([p_i_1]), p_i) ** 2
    dis_ca = dis(p_i, np.array([p_i_2])) ** 2
    return arccos((dis_ab + dis_bc - dis_ca) / (dis_ab + dis_bc)) - theta


# Given the number of sides of the polygon and the distance between the rover and the fish
# (radius), we calculate the angle and side length of the corresponding regular polygon.
def polygon(sides, radius=1, rotation=0, translation=None):
    one_segment = np.pi * 2 / sides

    points = [
        (np.sin(one_segment * i + rotation) * radius,
         np.cos(one_segment * i + rotation) * radius, 0)
        for i in range(sides)]

    if translation:
        points = [[sum(pair) for pair in zip(point, translation)]
                  for point in points]

    points = np.array(points)
    side_length = norm(points[0] - points[1])
    theta = ((sides - 2) * np.pi) / sides
    return side_length, theta


# Calculate the next best position for the Rover given constraints
def objective_function(x, rov_lst, p_t, mu_lst, radius_length, n_polygon):
    p_i_2, p_i_1 = rov_lst
    p_i = x
    side_length, theta = polygon(sides=n_polygon, radius=radius_length)

    f1 = mu_lst[0] * (equal_radius(p_i, p_i_1, p_t) ** 2)
    f2 = mu_lst[1] * (fixed_distance(p_i, p_i_1, side_length) ** 2)
    f3 = mu_lst[2] * (fixed_radius(p_i, p_t, radius_length) ** 2)
    f4 = mu_lst[3] * (fixed_angle(p_i, p_i_1, p_i_2, theta) ** 2)

    return f1 + f2 + f3 + f4


# Creates a list of locations around the zero point of all
# potential locations for the next Rover location
def create_heat_map(center_point):
    xs = np.arange(center_point[0] - 70, center_point[0] + 70, 0.5)
    ys = np.arange(center_point[1] - 70, center_point[1] + 70, 0.5)
    xx, yy = np.meshgrid(xs, ys, sparse=False)
    n = len(np.ndarray.flatten(xx))

    x = np.zeros((n, 3))
    x[:, 0] = np.ndarray.flatten(xx)
    x[:, 1] = np.ndarray.flatten(yy)
    x[:, 2] = 0  # 0 - 0.01 * randn(n)

    return x


# Given the last two Rover locations, we calculate the next best location (Given the type of polygon).
# This function can also calculate the next additional steps for the rover assuming the target is not moving.
def next_location(rov_lst, fish_loc, radius_length, n_polygon, n_steps=1):
    point_lst = np.copy(rov_lst)
    fish_loc = np.copy(fish_loc)
    next_location_lst = []
    x = create_heat_map(fish_loc)
    mu_lst = np.array([0.015, 0.07, 0.015, 0.9])
    # mu_lst = np.array([0.0, 0.07, 0.03, 0.9])

    for i in range(n_steps):
        objective_func_mat = objective_function(x=x,
                                                rov_lst=point_lst,
                                                p_t=fish_loc,
                                                mu_lst=mu_lst,
                                                radius_length=radius_length,
                                                n_polygon=n_polygon)

        res = x[np.argmin(objective_func_mat)]
        point_lst[0] = point_lst[1]
        point_lst[1] = res
        next_location_lst.append(res)

    return np.array(next_location_lst)

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

if __name__ == '__main__':
    num_file = 1663571619
    n_polygon = 4
    n_steps = 10
    path = f'{parent_dir}/plots/tests/test_' + str(num_file)
    path2 = f'{parent_dir}/data'

    if os.path.isdir(path + "/next_location_test"):
        shutil.rmtree(path + "/next_location_test")
    os.mkdir(path + "/next_location_test")

    # with open(path2 + '/debug/NextLocation' + str(num_file) + '.pkl', 'rb') as f:
    #     next_location_lst = pickle.load(f)

    with open(path + '/data_output' + str(num_file) + '.pkl', 'rb') as run_data_output:
        rover_lst, t_lst, fish_lst, idx_in_use_lst, utm_x, utm_y, target_real_loc, rpi_target_real_loc, _, _, _, _, _ = pickle.load(run_data_output)

    n = len(t_lst)
    rover_lst = np.array(rover_lst)
    t_lst = np.array(t_lst)
    fish_lst = np.array(fish_lst)
    idx_in_use_lst = np.array(idx_in_use_lst)
    # next_location_lst[3:] = next_location_lst[3:] - np.array([utm_x, utm_y])
    target_real_loc = target_real_loc - np.array([utm_x, utm_y, 0])

    x0, y0 = np.mean(target_real_loc, axis=0)[0:2]  # next_location_lst[7][0:2]
    rover_lst = rover_lst - np.array([x0, y0, 0])
    fish_lst = fish_lst - np.array([x0, y0, 0])
    # next_location_lst = next_location_lst - np.array([x0, y0])
    target_real_loc = target_real_loc - np.array([x0, y0, 0])
    rpi_target_real_loc = rpi_target_real_loc - np.array([x0, y0, 0])

    fig, ax = plt.subplots(1, 1, figsize=(16.0, 10.0))

    for i in np.arange(start=n_polygon, stop=n, step=1):
        last_fish_loc = fish_lst[i - n_polygon]
        if target_real_loc.shape[0]==1:
            real_fish_loc = target_real_loc[0]
        else:
            real_fish_loc = target_real_loc[i - n_polygon]
        next_point = next_location(rov_lst=rover_lst[i - 2:i],
                                   fish_loc=np.copy(last_fish_loc),
                                   radius_length=8,
                                   n_polygon=n_polygon,
                                   n_steps=n_steps)

        ax.set_xlim([-70, 70])
        ax.set_ylim([-70, 70])
        # last points of the rover
        ax.plot(rover_lst[i - n_polygon:i, 0], rover_lst[i - n_polygon:i, 1], c='red')
        ax.scatter(last_fish_loc[0], last_fish_loc[1], c='royalblue')
        ax.scatter(real_fish_loc[0], real_fish_loc[1], c='red')

        # where the rover went in real next point
        ax.plot([rover_lst[i - 1, 0], rover_lst[i, 0]], [rover_lst[i - 1, 1], rover_lst[i, 1]], c='pink')
        # where the rover shold go by rover algo
        # ax.plot([rover_lst[i - 1, 0], next_location_lst[i - 1][0]], [rover_lst[i - 1, 1], next_location_lst[i - 1][1]],
        #         c='blue')
        # next points of the rover algo simulate in this pc
        ax.plot([rover_lst[i - 1, 0], next_point[0, 0]], [rover_lst[i - 1, 1], next_point[0, 1]], c='green')
        if n_steps > 1:
            ax.plot(next_point[:n_steps, 0], next_point[:n_steps, 1], c='green')

        fig.show()
        plt.savefig(path + '/next_location_test/img' + str(1000 + i) + '.png', dpi=150)
        ax.cla()

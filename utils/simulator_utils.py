import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import rand, randn
from numpy import floor, abs, sqrt, append
from matplotlib_scalebar.scalebar import ScaleBar

import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))


# This function creates a random trajectory for the target with preferred directionality.
def rand_path(start_location, maxIt, length):
    count = 0
    path_list = np.array([start_location, start_location, start_location])
    new_location = np.array([0, 0, 0])

    while count < maxIt:
        # Each time we choose a number of steps, we are interested in going in a particular direction
        n_steps = int(floor(abs(randn(1) * 10)) + 1)
        # Each time we choose a direction in which we are interested, the target will move.
        k = abs(rand(1))
        for j in range(n_steps):

            if count >= maxIt:
                return path_list

            # Each time the target can move in a straight line or on a slope.
            z = abs(rand(1))

            if k <= 0.6 and z <= 0.8:
                new_location[0] = path_list[-1, 0] + (length * 0.5) * z + (length * 0.2)
                new_location[1] = 0 - sqrt(length ** 2 - (new_location[0] - path_list[-1, 0]) ** 2) + path_list[-1, 1]

            elif k <= 0.6 and z > 0.8:
                new_location[0] = path_list[-1, 0] + length
                new_location[1] = path_list[-1, 1]

            elif k > 0.6 and z <= 0.8:
                new_location[0] = path_list[-1, 0] - (length * 0.5) * z + (length * 0.2)
                new_location[1] = 0 + sqrt(length ** 2 - (new_location[0] - path_list[-1, 0]) ** 2) + path_list[-1, 1]

            elif k > 0.6 and z > 0.8:
                new_location[0] = path_list[-1, 0] - length
                new_location[1] = path_list[-1, 1]

            new_location[2] = path_list[-1, 2] - abs(randn(1)) / 10
            path_list = append(path_list, np.array([new_location]), axis=0)
            count += 1

    return path_list


# Deletes results of previous runs .
def delete_old_files(start_path):
    if os.path.isdir(start_path + "plots/temp_images"):
        shutil.rmtree(start_path + "plots/temp_images")
    os.mkdir(start_path + "plots/temp_images")

    if os.path.exists(start_path + "plots/movie_sim.mp4"):
        os.remove(start_path + "plots/movie_sim.mp4")


# Deletes results of previous runs.
def delete_old_files(start_path):
    path = start_path + "plots/temp_images"
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.mkdir(path)

    path = start_path + "plots/movie_sim.mp4"
    if os.path.exists(path):
        os.remove(path)


def error_fn(sigma, size):
    if size == 1:
        error_val = np.random.normal(0, sigma, 1)
    else:
        error_val = np.zeros(size)
        error_val[0:size-1] = np.random.normal(0, sigma, size-1)
    return error_val


def create_regular_polygon_vertexs(sides, r, center_point_x=0, center_point_y=0, rotate=0):
    """ args:
            sides (int): Number of polygon sides.
            center_point [x,y] (float): Coordinates of center point.
            r (float): Radius (diameter = 2 x r + 1.).
            rotate (float): Rotation in degrees relative to origin. """
    coords = []
    theta = np.radians(rotate)
    n = sides + 1
    for s in range(n):
        t = 2.0 * np.pi * s / sides + theta
        coords.append([int(r * np.cos(t) + center_point_x),
                       int(r * np.sin(t) + center_point_y)])
    vertexs = np.array(coords[0:n - 1])
    return vertexs


def check_polygon_length(sides, r):
    sides = 3 if sides == 'triangle_only' else sides
    vertexs = create_regular_polygon_vertexs(sides, r)
    side_length = np.linalg.norm(vertexs[0] - vertexs[1])
    return side_length


def plot_target_path(target_location_lst, path):
    fig, ax = plt.subplots(figsize=(16.0, 16.0))
    ax.set_box_aspect(1)
    ax.plot(target_location_lst[:, 0], target_location_lst[:, 1], linewidth=5, zorder=-1)
    # ax.set_xlabel('x (meter)')
    # ax.set_ylabel('y (meter)')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.scatter(target_location_lst[0, 0], target_location_lst[0, 1], c='g', s=350)
    ax.text(target_location_lst[0, 0] - 5, target_location_lst[0, 1] - 8, 'Start', fontsize=24)
    ax.scatter(target_location_lst[-1, 0], target_location_lst[-1, 1], c='r', s=350)
    ax.text(target_location_lst[-1, 0] - 3, target_location_lst[-1, 1] - 8, 'End', fontsize=24)
    scale_bar = ScaleBar(1, location="upper right", width_fraction=0.02, pad=0.8, border_pad=0.8, length_fraction=0.25, box_alpha=0.0)
    ax.add_artist(scale_bar)
    ax.set_axis_off()
    fig.show()
    fig.savefig(path, dpi=600, transparent=True)


def plot_simulator_with_receiver(rover_location_lst, target_location_lst, target_calc_lst, time_lst, maxIter, num_hyper, start_path):
    idx = target_location_lst.shape[0] - 6
    fig, ax = plt.subplots(figsize=(16.0, 10.0))
    ax.set_box_aspect(1)
    rover_location_lst = rover_location_lst[idx: idx + 4]
    target_location_lst = target_location_lst[idx-5: idx + 4]
    target_calc_lst = target_calc_lst[idx: idx + 4]
    time_lst = time_lst[idx: idx + 4]
    ax.clear()
    colors_lst = [['g', 'black'], ['orange', 'black'], ['purple', 'black']]
    text_lst = [['[R1,R2]'], ['[R2,R3]'], ['[R3,R4]']]
    reciver_names_lst = [["R1", "R2"], ["R2", "R3"], ["R3", "R4"]]

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_axis_off()

    ax.scatter(target_calc_lst[-1, 0], target_calc_lst[-1, 1], color='pink', s=150, label='estimate location')
    ax.scatter(target_location_lst[-1, 0], target_location_lst[-1, 1], color='r', s=150, label='real location')
    ax.plot(target_location_lst[:, 0], target_location_lst[:, 1], linestyle='--',
            linewidth=2, zorder=-1, color='gray')

    for i in range(3):
        last_rover_location_lst = rover_location_lst[i: i + 2]

        from utils.time_utils import calculate_time_difference
        dt = calculate_time_difference(time_lst[i:i+2])

        from main.simulator import plot_calc
        plot_calc(ax, fig, last_rover_location_lst, dt, 1, x_lim=None, y_lim=None, show_title=False,
                  color=colors_lst[i], text=text_lst[i], style_lst=['solid'],
                  reciver_names=reciver_names_lst[i])

        ax.text(last_rover_location_lst[0, 0] + 1.1, last_rover_location_lst[0, 1] - 0.5,
                reciver_names_lst[i][0], fontsize=20, zorder=2)
        ax.text(last_rover_location_lst[1, 0] + 1.1, last_rover_location_lst[1, 1] - 0.5,
                reciver_names_lst[i][1], fontsize=20, zorder=2)

        ax.set_xlim((np.mean(rover_location_lst[:, 0]) - 20, np.mean(rover_location_lst[:, 0]) + 20))
        ax.set_ylim((np.mean(rover_location_lst[:, 1]) - 20, np.mean(rover_location_lst[:, 1]) + 20))

    from matplotlib_scalebar.scalebar import ScaleBar
    scale_bar = ScaleBar(1, location="lower right")
    ax.add_artist(scale_bar)
    ax.legend(loc=3)  # ax.legend(fontsize=13, loc=3, facecolor='whitesmoke', edgecolor='white')

    print("123")
    # fig.savefig(start_path + 'plots/temp_images/sim' + str(1000 + i) + '.png', dpi=150)

    return fig, ax


def plot_complete_simulator(rover_location_lst, target_location_lst, target_calc_lst):
    fig, ax = plt.subplots(figsize=(16.0, 16.0))
    ax.set_box_aspect(1)
    ax.clear()
    ax.set_box_aspect(1)
    ax.plot(target_location_lst[:, 0], target_location_lst[:, 1], label="target path", c='b')
    ax.plot(rover_location_lst[:, 0], rover_location_lst[:, 1], label="target path", c='g')
    ax.plot(target_calc_lst[:, 0], target_calc_lst[:, 1], label="target path", c='r')
    ax.set_xlim((np.min(target_location_lst[:, 0]) - 20, np.max(target_location_lst[:, 0]) + 20))
    ax.set_ylim((np.min(target_location_lst[:, 1]) - 20, np.max(target_location_lst[:, 1]) + 20))
    scale_bar = ScaleBar(1, location="lower right")
    ax.add_artist(scale_bar)
    fig.show()

    return fig, ax


def plot_animator_simulator(rover_location_lst, target_location_lst, target_calc_lst, n_steps=5, is_end=False):

    if is_end:
        rover_location_lst = rover_location_lst[-30:]
        target_location_lst = target_location_lst[-30:]
        target_calc_lst = target_calc_lst[-30:]

    fig, ax = plt.subplots(figsize=(16.0, 16.0))
    ax.set_box_aspect(1)
    ax.clear()
    ax.set_box_aspect(1)
    plt.pause(0.1)
    label_lst = ["target path", "rover last locations", "target estimate location"]

    for i in range(rover_location_lst.shape[0]-n_steps):
        ax.plot(target_location_lst[:i+n_steps, 0], target_location_lst[:i+n_steps, 1], label=label_lst[0], c='b', linewidth=4)
        ax.plot(rover_location_lst[i:i+n_steps, 0], rover_location_lst[i:i+n_steps, 1], label=label_lst[1], c='g', linewidth=2)
        ax.scatter(target_calc_lst[i+n_steps, 0], target_calc_lst[i+n_steps, 1], label=label_lst[2], c='r', s=150)
        ax.set_xlim((np.min(target_location_lst[:, 0]) - 20, np.max(target_location_lst[:, 0]) + 20))
        ax.set_ylim((np.min(target_location_lst[:, 1]) - 20, np.max(target_location_lst[:, 1]) + 20))
        plt.xlabel("X axis [m]")
        plt.ylabel("Y axis [m]")
        # scale_bar = ScaleBar(1, location="lower right")
        # ax.add_artist(scale_bar)
        ax.legend(loc=0)
        fig.show()
        fig.savefig(f'{parent_dir}/test/plot_sim' + str(100 + i) + '.png', dpi=150, transparent=True)
        plt.pause(0.1)
        ax.clear()
        ax.set_box_aspect(1)

    print("123")
    return None


if __name__ == '__main__':
    s = check_polygon_length(4, 7)
    print(s)
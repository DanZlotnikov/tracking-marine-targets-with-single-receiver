import os
import numpy as np
import plotly.express as px
from numpy.random import randn
from numpy import dot
from utils.time_utils import calculate_time_difference


# We will mark the following parameters:
# p_{i}, p_{i-1}, p_{i-2} = p_i, p_i_1, p_i_2

def dis(p_a, p_b):
    return ((p_a[:, 0].T - p_b[:, 0]) ** 2 +
            (p_a[:, 1].T - p_b[:, 1]) ** 2 +
            (p_a[:, 2].T - p_b[:, 2]) ** 2) ** 0.5


def hyperbola(p_t, p_a, p_b, range_dif):
    return np.abs(dis(p_t, p_a) - dis(p_t, p_b) - range_dif) ** 1


def objective_function(p_t, p_a, p_b, range_dif, mu_lst, show_separate_map=None):
    function_val = (mu_lst * (hyperbola(p_t, p_a, p_b, range_dif) ** 1)).T
    if show_separate_map is None:
        return sum(function_val)
    else:
        return function_val[show_separate_map]


# Creates a list of locations around the zero point of all
# potential locations for the next Rover location
def create_heat_map(center_point, size_heat_map, size_divide):
    xs = np.arange(center_point[0] - size_heat_map, center_point[0] + size_heat_map, 1 / size_divide)
    ys = np.arange(center_point[1] - size_heat_map, center_point[1] + size_heat_map, 1 / size_divide)
    xx, yy = np.meshgrid(xs, ys, sparse=False)
    n = len(np.ndarray.flatten(xx))

    x = np.zeros((n, 3))
    x[:, 0] = np.ndarray.flatten(xx)
    x[:, 1] = np.ndarray.flatten(yy)
    x[:, 2] = 0  # 0 - 0.01 * randn(n)

    return x


# using plotly, we plot the heatmap
def plot_heat_map(objective_func_heat_map, idx, max_val, start_path):
    if idx is None:
        pass
    else:
        fig = px.imshow(objective_func_heat_map / max_val,
                        labels=dict(x="Location (m)", y="Location (m)", color="Position Probability"))
        path = os.path.abspath("plots/temp_images") + '/'

        if (type(idx) == str) & (idx == "test"):
            fig.show()
        elif type(idx) != str:
            path = os.path.abspath(start_path + "/heatmap_images") + '/'
            if not os.path.isdir(path):
                os.mkdir(path)
            fig.write_html(path + "file" + str(idx + 1000) + ".html")


# Given the last two Rover locations, we calculate the next best location (Given the type of polygon).
# This function can also calculate the next additional steps for the rover assuming the target is not moving.
def tdoa_heat_map(rov_lst, p_t, idx=None, size_heat_map=70, size_divide=2, show_separate_map=None, start_path=None):
    rov_lst = np.array(rov_lst)
    n_points = rov_lst.shape[0]
    velocity = 1503

    mu_lst = np.ones(n_points - 1) / (n_points - 1)
    time_dif = calculate_time_difference(p_t)
    range_dif = time_dif * velocity
    # range_dif = (p_t[1:n_points] - p_t[0:n_points - 1]) * velocity

    p_b = rov_lst[0:n_points - 1]
    p_a = rov_lst[1:n_points]

    mean_rov = np.mean(rov_lst, axis=0)
    x = create_heat_map(mean_rov, size_heat_map, size_divide)
    pt_lst = np.swapaxes(np.repeat(np.array([x]), n_points - 1, axis=0), 2, 1)

    objective_func_mat = objective_function(pt_lst, p_a, p_b, range_dif, mu_lst, show_separate_map)

    objective_func_heat_map = np.reshape(objective_func_mat, (dot(size_heat_map, 2 * size_divide),
                                                              dot(size_heat_map, 2 * size_divide)))
    res = x[np.argmin(objective_func_mat)]

    plot_heat_map(objective_func_heat_map, idx, np.max(objective_func_mat), start_path)

    return np.array([res]), False


if __name__ == '__main__':
    last_locations = np.array([[-0.91077861, 8.18877979, 0.],
                               [-4.65233306, -11.9388903, 0.],
                               [8.26671784, -1.4158498, 0.]])

    last_times = np.array([2.609, 2.62, 2.609])

    res, _ = tdoa_heat_map(rov_lst=last_locations,
                           p_t=last_times,
                           idx="test",
                           size_heat_map=30,
                           size_divide=2,
                           show_separate_map=None,
                           start_path=None)

    print(res)

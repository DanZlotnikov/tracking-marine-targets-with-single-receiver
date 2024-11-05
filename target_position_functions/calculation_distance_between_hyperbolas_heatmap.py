from __future__ import division

import os
import pickle
import numpy as np
import plotly.express as px
from numpy.linalg import norm
from utils.quartic_solver_utils import quartic_roots, ferrari_quartic_solver
from numpy import power, divide, isreal, isfinite, roots, real, dot, multiply
from utils.time_utils import calculate_time_difference


# Creates a list of locations around the zero point of all
# potential locations for the next Rover location
def create_heat_map(center_point, size_heat_map, size_divide):
    xs = np.arange(center_point[0] - size_heat_map, center_point[0] + size_heat_map, 1.0 / size_divide)
    ys = np.arange(center_point[1] - size_heat_map, center_point[1] + size_heat_map, 1.0 / size_divide)
    n = 2 * size_heat_map * size_divide
    xs, ys = xs[:n], ys[:n]

    xx, yy = np.meshgrid(xs, ys, sparse=False)
    n = len(np.ndarray.flatten(xx))

    x = np.zeros((n, 3))
    x[:, 0] = np.ndarray.flatten(xx)
    x[:, 1] = np.ndarray.flatten(yy)
    x[:, 2] = 0

    return x


# calculates the quartic coefficients for distance between the hyperbola and a point.
def quartic_coefficients_old(A, B, C, a, b):
    k = power(a, 2) + power(b, 2)
    A_tag = 0 + np.array([power(k, 2)]).T + dot(A, 0)
    B_tag = 0 - dot(2, multiply(B, np.array([multiply(np.power(b, 2), k)]).T))
    C_tag = 0 + multiply(np.power(B, 2), np.array([power(b, 4)]).T) + \
            np.array([multiply(np.power(b, 2), power(k, 2))]).T
    D_tag = 0 - dot(2, multiply(B, np.array([multiply(power(b, 4), k)]).T))
    E_tag = 0 + multiply(power(B, 2), np.array([power(b, 6)]).T) - \
            multiply(power(A, 2), np.array([multiply(power(a, 2), power(b, 2))]).T)
    return A_tag, B_tag, C_tag, D_tag, E_tag


# calculates the quartic coefficients for distance between the hyperbola and a point.
def quartic_coefficients(A, B, C, a, b, depth=0):
    z = np.array([1.0 + (depth ** 2) / (val ** 2) if not np.isnan(val) else 1.0 for val in b])
    k = power(a, 2) + power(b, 2)
    A_tag = 0 + np.array([power(k, 2)]).T + dot(A, 0)
    B_tag = 0 - dot(2, multiply(B, np.array([multiply(np.power(b, 2), k)]).T))
    C_tag = 0 + multiply(np.power(B, 2), np.array([power(b, 4)]).T) + \
            multiply(np.array([multiply(np.power(b, 2), power(k, 2))]).T, np.array([z]).T) - \
            multiply(power(A, 2), np.array([multiply(power(a, 2), power(b, 2))]).T)
    D_tag = 0 - dot(2, multiply(B, np.array([multiply(power(b, 4), multiply(k, z))]).T))
    E_tag = 0 + multiply(power(B, 2), multiply(np.array([power(b, 6)]), np.array([z])).T)

    return A_tag, B_tag, C_tag, D_tag, E_tag


def start_algo(a, b, points, NRD, c_lst, quartic_solver_method='fqs', depth=0):
    A, B, C = points[:, 0, :], points[:, 1, :], points[:, 2, :]
    A_tag, B_tag, C_tag, D_tag, E_tag = quartic_coefficients(A, B, C, a, b, depth)

    equ_param = np.transpose(np.array([A_tag.T, B_tag.T, C_tag.T, D_tag.T, E_tag.T]), (2, 1, 0))
    check_is_real = isreal(equ_param).all()
    check_in_finite = (np.isnan(equ_param) | isfinite(equ_param)).all()

    if check_is_real & check_in_finite:
        pass
    else:
        return np.array([])

    if quartic_solver_method == 'numpy':
        distance_res = numpy_root_solver_fn(equ_param, A, B, C, NRD, a, b, c_lst)
    else:
        distance_res = quartic_solver_fn(equ_param, A, B, C, NRD, a, b, c_lst,
                                         quartic_solver_method=quartic_solver_method)

    return distance_res


def quartic_solver_fn(equ_param, A, B, C, NRD, a, b, c_lst, quartic_solver_method='fqs'):
    dis_res = np.zeros((equ_param.shape[0], equ_param.shape[1]), dtype=float)
    siman = NRD
    for i in range(equ_param.shape[0]):
        sim = 1
        if siman[i] < 0:
            sim = -1

        if a[i] == 0:
            dis_min = np.sqrt((0 - np.array([A[i]]).T) ** 2 +
                              0 ** 2 +
                              (0 - np.array([C[i]]).T) ** 2).T[0]
        elif (np.abs(NRD[i]) >= 0.9) | (b[i] == np.nan):
            new_point = np.array([c_lst[i] * sim, 0])
            dis_min1 = np.sqrt((new_point[0] - np.array([A[i]]).T) ** 2 +
                               (new_point[1] - np.array([B[i]]).T) ** 2 +
                               (0 - np.array([C[i]]).T) ** 2).T[0]
            dis_min2 = np.sqrt(0 ** 2 +
                               (0 - np.array([B[i]]).T) ** 2 +
                               (0 - np.array([C[i]]).T) ** 2).T[0]
            if sim > 0:
                dis_min = (A[i] >= new_point[0]) * dis_min2 + (A[i] < new_point[0]) * dis_min1
            else:
                dis_min = (A[i] <= new_point[0]) * dis_min2 + (A[i] > new_point[0]) * dis_min1

        else:
            equ_param_i = equ_param[i].T
            y = np.zeros(1)
            if quartic_solver_method == 'ferrari2':
                y = ferrari_quartic_solver(equ_param_i[0], equ_param_i[1], equ_param_i[2],
                                           equ_param_i[3], equ_param_i[4], ver='v2')
            elif quartic_solver_method == 'ferrari':
                y = ferrari_quartic_solver(equ_param_i[0], equ_param_i[1], equ_param_i[2],
                                           equ_param_i[3], equ_param_i[4], ver='v1')
            else:  # quartic_solver_method == 'fqs':
                y = quartic_roots(equ_param_i.T)

            y = np.where(y.imag != 0, np.inf, y.real)
            x = multiply(divide(a[i], b[i]), np.sqrt(power(y, 2) + power(b[i], 2)))
            x = dot(sim, x)

            dis_all_option = np.sqrt((x - np.array([A[i]]).T) ** 2 +
                                     (y - np.array([B[i]]).T) ** 2 +
                                     (0 - np.array([C[i]]).T) ** 2)

            dis_min = np.min(dis_all_option, axis=1)
            if dis_min[dis_min == np.inf].shape[0] > 0:
                dis_min[dis_min == np.inf] = np.amax(dis_min[dis_min != np.inf])

        dis_res[i] = dis_min

    return dis_res


def numpy_root_solver_fn(equ_param, A, B, C, NRD, a, b, c_lst):
    dis_res = np.zeros((equ_param.shape[0], equ_param.shape[1]), dtype=float)
    siman = NRD
    for i in range(equ_param.shape[0]):
        sim = 1
        if siman[i] < 0:
            sim = -1

        if a[i] == 0:
            dis_res[i] = np.sqrt((0 - np.array([A[i]]).T) ** 2 + 0 ** 2 + (0 - np.array([C[i]]).T) ** 2).T[0]
        elif (np.abs(NRD[i]) >= 0.9) | (b[i] == np.nan):
            new_point = np.array([c_lst[i] * sim, 0])
            dis_min1 = np.sqrt((new_point[0] - np.array([A[i]]).T) ** 2 +
                               (new_point[1] - np.array([B[i]]).T) ** 2 +
                               (0 - np.array([C[i]]).T) ** 2).T[0]
            dis_min2 = np.sqrt(0 ** 2 +
                               (0 - np.array([B[i]]).T) ** 2 +
                               (0 - np.array([C[i]]).T) ** 2).T[0]
            if sim > 0:
                dis_min = (A[i] >= new_point[0]) * dis_min2 + (A[i] < new_point[0]) * dis_min1
            else:
                dis_min = (A[i] <= new_point[0]) * dis_min2 + (A[i] > new_point[0]) * dis_min1
            dis_res[i] = dis_min
        else:
            for j in range(equ_param.shape[1]):
                point_j_for_hyperbola_i = equ_param[i, j]
                y = roots(point_j_for_hyperbola_i)
                y = real(y[y.imag == 0])

                if y.shape[0] > 0:
                    x = multiply(divide(a[i], b[i]), np.sqrt(power(y, 2) + power(b[0], 2)))
                    x = multiply(sim, x)

                    dis_all_option = np.sqrt((x - np.array([A[i, j]]).T) ** 2 +
                                             (y - np.array([B[i, j]]).T) ** 2 +
                                             (0 - np.array([C[i, j]]).T) ** 2)

                    dis_min = np.min(dis_all_option)
                    dis_res[i, j] = dis_min

                elif j >= 1:
                    dis_res[i, j] = dis_res[i, j - 1]
                else:
                    dis_res[i, j] = 10e5

    return dis_res


def calc_parm(p1, p2, RD):
    c = norm(p1 - p2, axis=1) / 2
    NRD = RD / (2 * c)
    Ax = abs(RD) / 2
    By2 = power(c, 2) - power(Ax, 2)
    By2 = np.where(By2 > 0, By2, np.nan)
    By = np.sqrt(By2)

    return Ax, By, NRD, c


# using plotly, we plot the heatmap
def plot_heat_map(all_dis_heat_map, idx, start_path):
    if idx is None:
        pass
    else:
        fig = px.imshow(all_dis_heat_map / np.amax(all_dis_heat_map),
                        labels=dict(x="Location (m)", y="Location (m)", color="Position Probability"))

        if (type(idx) == str) & (idx == "test"):
            fig.show()
        elif type(idx) != str:
            path = os.path.abspath(start_path + "/heatmap_images") + '/'
            if not os.path.isdir(path):
                os.mkdir(path)
            fig.write_html(path + "file" + str(idx + 1000) + ".html")


def heat_map_hyperbola_distance(rov_lst, p_t, size_heat_map, size_divide=2, idx=None, show_separate_map=None,
                                quartic_solver_method='fqs', start_path=None, min_val=None, depth=None,
                                is_3d=False):

    if (depth is None) or (not is_3d):
        depth = 0

    rov_lst = np.array(rov_lst)
    n_points = rov_lst.shape[0]
    velocity = 1503

    mu_lst = divide(np.ones(n_points - 1), (n_points - 1))
    time_dif = calculate_time_difference(p_t)
    range_dif = time_dif * velocity
    # range_dif = dot(p_t[1:n_points] - p_t[0:n_points - 1], velocity)

    p_b = rov_lst[0:n_points - 1]
    p_a = rov_lst[1:n_points]

    Ax_lst, By_lst, NRD, c_lst = calc_parm(p_a, p_b, range_dif)

    theta = np.arctan2(p_a[:, 1] - p_b[:, 1], p_a[:, 0] - p_b[:, 0])

    for i in range(n_points - 1):
        if Ax_lst[i] != 0:
            theta[i] = np.pi + theta[i]

    centers = np.dot(0.5, p_a + p_b)

    mean_rov = np.mean(rov_lst, axis=0)
    x = create_heat_map(mean_rov, size_heat_map, size_divide)

    repeat_points = np.repeat(np.array([x]), n_points - 1, axis=0)
    for i in range(n_points - 1):
        R = np.array([[np.cos(theta[i]), 0 - np.sin(theta[i])], [np.sin(theta[i]), np.cos(theta[i])]])
        repeat_points[i, :, 0:2] = np.matmul((repeat_points[i, :, 0:2] - centers[i, 0:2]), R)
    pt_lst = np.swapaxes(repeat_points, 2, 1)

    min_hyper = start_algo(Ax_lst, By_lst, pt_lst, NRD, c_lst, quartic_solver_method, depth)

    heat_maps = np.reshape(min_hyper, (min_hyper.shape[0],
                                       dot(size_heat_map, 2 * size_divide),
                                       dot(size_heat_map, 2 * size_divide)))

    if show_separate_map is None:
        all_dis_heat_map = sum(heat_maps)
    else:
        # all_dis_heat_map = heat_maps[show_separate_map]
        all_dis_heat_map = sum([heat_maps[k] for k in show_separate_map])

    plot_heat_map(np.copy(all_dis_heat_map), idx, start_path)

    all_dis_heat_vector = np.reshape(all_dis_heat_map, (power(dot(size_heat_map, 2 * size_divide), 2)))

    if min_val is not None:
        if np.min(all_dis_heat_vector) > min_val:
            return np.array([]), True

    res = x[np.argmin(all_dis_heat_vector)]
    res[2] = depth
    return np.array([res]), False

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

if __name__ == '__main__':
    num_file = 1663683428
    check_idx_test = 445
    show_separate_map = None
    start_path = '{parent_dir}/plots/tests/test_' + str(num_file) + '/'
    file_name = 'data_output' + str(num_file) + '.pkl'
    with open(start_path + file_name, "rb") as run_data_output:
        data = pickle.load(run_data_output)
        rover_lst, t_lst, fish_lst, idx_in_use_lst, x0, y0, _, _, _, _, _, _, _ = data

    last_locations = []
    last_times = []
    for i, idx in enumerate(idx_in_use_lst):
        if len(last_locations) >= 4:
            last_locations.pop(0)
            last_times.pop(0)

        last_locations.append(rover_lst[i])
        last_times.append(t_lst[i])

        if idx == check_idx_test:
            if True and len(last_locations) >= 4:
                res, _ = heat_map_hyperbola_distance(rov_lst=np.array(last_locations),
                                                     p_t=np.array(last_times),
                                                     quartic_solver_method='fqs',
                                                     size_heat_map=50,
                                                     size_divide=4,
                                                     idx="test",
                                                     show_separate_map=show_separate_map,
                                                     start_path=None)
                print(res)

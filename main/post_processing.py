# data_output1659624344.pkl
import pickle
import numpy as np
from parameters import args
from numpy.linalg import norm
import matplotlib.pyplot as plt
from utils.main_utils import create_plot_and_map_final
from numpy import abs, mean, arange, argmin, floor, around, min
from utils.plots_utils import distance_error_plot, error_plot_for_list


def read_file(num_file):
    start_path = 'plots/tests/test_' + str(num_file)
    with open(start_path + '/data_output' + str(num_file) + '.pkl', 'rb') as run_data_output:
        rover_lst, t_lst, target_esti_lst, idx_in_use_lst, x0, y0, target_real_loc_lst, _, \
        check_poly_lst, map_mean_point, is_exits_real_target, angle_lst, snr_lst = pickle.load(run_data_output)

    rover_lst = np.array(rover_lst)
    t_lst = np.array(t_lst)
    target_esti_lst = np.array(target_esti_lst)
    idx_in_use_lst = np.array(idx_in_use_lst)
    target_real_loc_lst = np.array(target_real_loc_lst) - np.array([x0, y0, 0])
    check_poly_lst = np.array(check_poly_lst)
    angle_lst = np.array(angle_lst)
    snr_lst = np.array(snr_lst)

    return rover_lst, t_lst, target_esti_lst, idx_in_use_lst, target_real_loc_lst, \
           check_poly_lst, map_mean_point, is_exits_real_target, angle_lst, x0, y0, snr_lst


def calculate_time_dif_base(t_lst, idx_base, dist_normalize, velocity):
    time_mode10_lst = np.array(t_lst) % 10
    base_time = time_mode10_lst[idx_base] - dist_normalize / velocity
    t_lst_base = np.array([val if abs(val) < 1.0 else val + 10 * (val < 0) - 10 * (val >= 0)
                           for val in (time_mode10_lst - base_time)])
    return t_lst_base


def check_time_steps_and_estimate_dif(t_lst, target_real_loc_lst, target_esti_lst):
    min_len = np.min([len(t_lst) - 1, len(target_real_loc_lst), len(target_esti_lst) - 1])
    check_time_dif = t_lst[1:] - t_lst[0:-1] < 35.0
    check_target_dis_dif = norm(target_esti_lst[1:] - target_esti_lst[0:-1], axis=1) <= 10
    check_time_and_dis = [check_time_dif[i] and check_target_dis_dif[i] for i in range(min_len)]

    sss = norm(target_esti_lst[1:] - target_esti_lst[0:-1], axis=1)
    sss1 = t_lst[1:] - t_lst[0:-1]

    return check_time_and_dis

def calculate_angle(a, b, c):
    from target_position_functions.calculation_tdoa_classic import Triangular_rotation
    ab, ac, bc = norm(a - b), norm(a - c), norm(b - c)
    alpha = np.arccos((ab ** 2 + bc ** 2 - ac ** 2) / (2 * ab * bc))

    new_a, new_b, new_c = Triangular_rotation(np.array([a, b, c]))[0]
    siman = (new_a[1] >= 0) * 1 + (new_a[1] < 0) * -1

    return alpha * siman

def mean_post_processing(option, velocity, kernel_size,
                         rover_lst, t_lst, target_esti_lst, target_real_loc_lst):
    n = kernel_size
    check = norm(target_real_loc_lst[1:] - target_real_loc_lst[0:-1], axis=1)
    rover_lst = np.copy(rover_lst)
    t_lst = np.copy(t_lst)
    target_esti_lst = np.copy(target_esti_lst)
    target_real_loc_lst = np.copy(target_real_loc_lst)

    target_esti_lst_new = []  # [[0, 0, 0]] * 3
    for i in arange(start=2, stop=target_esti_lst.shape[0] - n, step=1):

        # take each time n points
        t_lst_n = t_lst[i:i + n]
        rover_lst_n = rover_lst[i:i + n]
        target_esti_lst_n = np.array(target_esti_lst[i:i + n])

        # calculate the location is the closest to the mean of the n estimates locations points
        mean_esti = mean(target_esti_lst_n, axis=0)
        idx = argmin(norm(target_esti_lst_n - mean_esti, axis=1))

        # calculate the vector direction between rover position and target estimate position
        dist_normalize = norm(rover_lst_n[idx] - target_esti_lst_n[idx])
        vec_direction_lst_n = np.array([val / norm(val) for val in (target_esti_lst_n - rover_lst_n)])
        t_lst_base = calculate_time_dif_base(t_lst_n, idx, dist_normalize, velocity)
        dist_correction = t_lst_base * velocity

        # calculate new estimate position by adding the correction vector to the rover position
        if option == 'linear filter kernel on spot':
            j = int(floor(n / 2))
            target_esti_lst[i + j] = rover_lst[i + j] + dist_correction[j] * vec_direction_lst_n[j]
        elif option == 'linear filter kernel on new array':
            j = int(floor(n / 2))
            target_esti_lst_new.append(rover_lst[i + j] + dist_correction[j] * vec_direction_lst_n[j])
        elif option == 'full linear filter kernel on spot':
            target_esti_lst[i:i + n] = rover_lst[i:i + n] + np.array([dist_correction]).T * vec_direction_lst_n

    esti_dis, target_real_loc_lst_new, t_lst_new = np.zeros(1), np.zeros(1), np.zeros(1)
    target_esti_lst = np.array(target_esti_lst)
    target_esti_lst_new = np.array(target_esti_lst_new)
    if option in ['linear filter kernel on spot', 'full linear filter kernel on spot']:
        target_esti_lst_new = target_esti_lst
        target_real_loc_lst_new = target_real_loc_lst[:-3]
        t_lst_new = t_lst[:-3]
    elif option in ['linear filter kernel on new array']:
        target_real_loc_lst_new = target_real_loc_lst[6:len(target_esti_lst_new) + 6]
        t_lst_new = t_lst[6:len(target_esti_lst_new) + 6]

    new_alpha_lst = []
    for i in np.arange(start=0, stop=len(target_esti_lst_new) - 4):
        a = target_esti_lst_new[i, 0:2]
        b = mean(rover_lst[i:i + 4, 0:2], axis=0)
        c = target_real_loc_lst[i, 0:2]
        alpha = calculate_angle(a, b, c) * 180 / np.pi
        new_alpha_lst.append(alpha)

    esti_dis = norm(target_real_loc_lst_new[:, 0:2] - target_esti_lst_new[:, 0:2], axis=1)
    esti_dis = around(esti_dis, decimals=2)
    print(esti_dis)
    print(mean(esti_dis))

    return target_esti_lst_new, target_real_loc_lst_new, t_lst_new, new_alpha_lst


def plot_results(num_file, target_real_lst, target_esti_lst, rover_lst, x0, y0, map_mean_point,
                 is_exits_real_target, angle_lst, t_lst, snr_lst, method_name, check_lst=None, lim=350, angle_bins=3):
    if check_lst is None:
        check_lst = np.array([True] * len(target_esti_lst))

    n_back_look = 0
    start_path = 'plots/tests/test_' + str(num_file) + '/analytics'
    full_path = start_path + '/' + method_name
    # start_path_rover_img = 'plots/tests/test_' + str(args.num_file) + '/rover_path_images'
    delete_and_create_folders(start_path, full_path)

    fig2, ax_map2, maps2, scale_bar2 = create_plot_and_map_final(args, map_mean_point)
    real_target_utm_lst = target_real_lst + np.array([x0, y0, 0])
    target_esti_utm_lst = target_esti_lst + np.array([x0, y0, 0])
    mean_point_utm = np.mean(np.array(rover_lst), axis=0)[0:2] + np.array([x0, y0])

    # At the end of the run, we present a wide picture of the map
    maps2.end_of_the_task(ax=ax_map2,
                          mean_point=mean_point_utm,
                          is_plot_real_target_location=is_exits_real_target,
                          target_real_lst=real_target_utm_lst[:len(check_lst)],
                          target_estimate_lst=target_esti_utm_lst[:len(check_lst)],
                          check_poly_lst=check_lst,
                          lim=lim)

    # plt.pause(2)
    plt.savefig(full_path + '/img_plot_summery.png', dpi=150)
    plt.close('all')

    if is_exits_real_target:
        plot_distance_error_plot(target_real_lst[:len(check_lst)], target_esti_lst[:len(check_lst)],
                                 check_lst, x0, y0, n_back_look, full_path)
    plot_angle_results(angle_lst[:len(check_lst)], full_path, check_lst, angle_bins=angle_bins)
    plot_time_steps_analytical(t_lst[:len(check_lst) + 1], full_path, filter_lst=check_lst)
    plot_dif_estimate_target_dis(target_esti_lst, full_path, filter_lst=check_lst)
    plot_snr_analytical(snr_lst, full_path, filter_lst=check_lst)
    plt.close('all')


def plot_distance_error_plot(target_real_lst, target_esti_lst, check_lst, x0, y0, n_back_look, start_path):
    distance_error_plot(target_esti_lst[:len(check_lst)], target_real_lst[:len(check_lst)],
                        check_lst, x0=x0, y0=y0, i=n_back_look, n_cut=0)
    plt.savefig(start_path + '/distance_error.png', dpi=150)
    # plt.pause(2)


def plot_angle_results(angle_lst, path, filter_lst=None, angle_bins=3):
    if filter_lst is None:
        filter_lst = np.array([True] * len(angle_lst))
    min_n = min([len(angle_lst), len(filter_lst)])
    angle_lst = angle_lst[:min_n]
    filter_lst = filter_lst[:min_n]
    bins = [-180, -160, -140, -120, -100, -80, -60, -40, -20, 0, 20, 40, 60, 80, 100, 120, 140, 160, 180]
    error_plot_for_list(angle_lst, filter_lst,
                        title="direction error",
                        unit_of_measurement='degrees',
                        bins=bins, ylim=[-180, 180], n_cut=0, is_polar=True, angle_bins=angle_bins)
    # direction_error_plot(angle_lst, filter_lst, n_cut=0)
    plt.savefig(path + '/direction_error.png', dpi=150)
    # plt.pause(2)



def plot_time_steps_analytical(t_lst, path, filter_lst=None):
    dt_steps_lst = t_lst[1:] - t_lst[0:-1]
    if filter_lst is None:
        filter_lst = np.array([True] * len(dt_steps_lst))
    min_n = min([len(dt_steps_lst), len(filter_lst)])
    dt_steps_lst = dt_steps_lst[:min_n]
    filter_lst = filter_lst[:min_n]
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    error_plot_for_list(dt_steps_lst, filter_lst,
                        title="Time difference in receiving transmitters",
                        unit_of_measurement='seconds',
                        bins=bins, ylim=[0, 100], n_cut=0)
    plt.savefig(path + '/time_steps_analytical.png', dpi=150)
    # plt.pause(2)


def plot_dif_estimate_target_dis(target_esti_lst, path, filter_lst=None):
    dt_steps_size = norm(target_esti_lst[1:] - target_esti_lst[0:-1], axis=1)
    if filter_lst is None:
        filter_lst = np.array([True] * len(dt_steps_size))
    min_n = min([len(dt_steps_size), len(filter_lst)])
    dt_steps_size = dt_steps_size[:min_n]
    filter_lst = filter_lst[:min_n]
    bins = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    error_plot_for_list(dt_steps_size, filter_lst,
                        title="Time difference in target estimation location",
                        unit_of_measurement='meters',
                        bins=bins, ylim=[0, 20], n_cut=0)
    plt.savefig(path + '/dif_estimate_target_dis.png', dpi=150)
    # plt.pause(2)


def plot_snr_analytical(snr_lst, path, filter_lst=None):
    if filter_lst is None:
        filter_lst = np.array([True] * len(snr_lst))
    min_n = min([len(snr_lst), len(filter_lst)])
    snr_lst = snr_lst[:min_n]
    filter_lst = filter_lst[:min_n]
    bins = [0, 10, 20, 30, 40, 50, 60]
    error_plot_for_list(snr_lst, filter_lst,
                        title="SNR",
                        unit_of_measurement='dB',
                        bins=bins, ylim=[0, 60], n_cut=0)
    plt.savefig(path + '/snr_analytical.png', dpi=150)
    # plt.pause(2)


def delete_and_create_folders(analytics_path, full_path):
    # check if analytics folder exits
    import os, shutil
    if not os.path.isdir(analytics_path):
        os.mkdir(analytics_path)

    # check if full_path is exits.
    if os.path.isdir(full_path):
        # If yes, delete it and create new one.
        shutil.rmtree(full_path)
    os.mkdir(full_path)


def save_to_csv(rover_lst, t_lst, target_esti_lst, target_real_lst, angle_lst, snr_lst):
    path = 'plots/tests/test_' + str(num_file) +'/data_table.csv'

    rover_lst = np.copy(rover_lst)
    t_lst = np.copy(t_lst)
    target_esti_lst = np.copy(target_esti_lst)
    target_real_lst = np.copy(target_real_lst)
    angle_lst = np.copy(angle_lst)
    snr_lst = np.copy(snr_lst)


    import pandas as pd
    data = []
    dif_n = rover_lst.shape[0] - target_esti_lst.shape[0]
    for i in range(len(rover_lst)):
        if i >= dif_n:
            data.append([list(rover_lst[i]), t_lst[i], list(target_esti_lst[i-dif_n]),
                         list(target_real_lst[i]), angle_lst[i-dif_n], snr_lst[i]])
        else:
            data.append([list(rover_lst[i]), t_lst[i], None, list(target_real_lst[i]), None, snr_lst[i]])

    pd.DataFrame(data=data,
                 columns=['rover location', 'time', 'estimate target location',
                          'real target location', 'angle', 'snr']).to_csv(path)
    print(123)



if __name__ == '__main__':
    option = {1: 'linear filter kernel on spot',
              2: 'full linear filter kernel on spot',
              3: 'linear filter kernel on new array'}

    option_check = {1: 'no post processing and no filter',
                    2: 'post process and no filter',
                    3: 'no post processing and polygon filter',
                    4: 'post process and time and dis filter',
                    5: 'no post processing and time and dis filter'}

    kernel_size = 9
    velocity = 1503.0
    # 1. 1657549631 2. 1659624344 3. 1663651329 4. 1663653769 5. 1663683428 6. 1663745933
    num_file = 1656829324
    idx_option = 1  # idx_option_check = 1

    angle_bins = 3
    total_plot_size = 450

    rover_lst, t_lst, target_esti_lst, idx_in_use_lst, target_real_lst, check_poly_lst, \
    map_mean_point, is_exits_real_target, angle_lst, x0, y0, snr_lst = read_file(num_file)

    if target_real_lst.shape[0] == 1:
        target_real_lst = np.array([list(target_real_lst[0])] * rover_lst.shape[0])

    save_to_csv(rover_lst, t_lst, target_esti_lst, target_real_lst, angle_lst, snr_lst)

    # if target_real_lst.shape[0] == 1:
    #     target_real_lst = np.array([list(target_real_lst[0])] * rover_lst.shape[0])

    new_target_esti_lst, new_target_real_lst, new_t_lst, new_alpha_lst = \
        mean_post_processing(option[idx_option], velocity, kernel_size,
                             rover_lst, t_lst, target_esti_lst, target_real_lst)

    # if method_name == 'no post processing and no filter':
    method_name = option_check[1]
    plot_results(num_file, target_real_lst[:len(target_esti_lst)], target_esti_lst,
                 rover_lst, x0, y0, map_mean_point, is_exits_real_target, angle_lst,
                 t_lst, snr_lst, method_name, lim=total_plot_size, angle_bins=angle_bins)

    # # elif method_name == 'post process and no filter':
    # method_name = option_check[2]
    # plot_results(num_file, new_target_real_lst, new_target_esti_lst, rover_lst, x0, y0,
    #              map_mean_point, is_exits_real_target, new_alpha_lst, new_t_lst,
    #              snr_lst, method_name, lim=total_plot_size, angle_bins=angle_bins)
    #
    # # elif method_name == 'no post processing and polygon filter':
    # method_name = option_check[3]
    # plot_results(num_file, target_real_lst[:-3], target_esti_lst, rover_lst, x0, y0,
    #              map_mean_point, is_exits_real_target, angle_lst, t_lst[:-3], snr_lst,
    #              method_name, check_poly_lst, lim=total_plot_size, angle_bins=angle_bins)

    # elif method_name == 'post process and time and dis filter':
    # method_name = option_check[4]
    # check_lst = check_time_steps_and_estimate_dif(new_t_lst, new_target_real_lst, new_target_esti_lst)
    # plot_results(num_file, new_target_real_lst, new_target_esti_lst, rover_lst, x0, y0, map_mean_point,
    #              is_exits_real_target, new_alpha_lst[:len(check_lst)], new_t_lst, snr_lst, method_name,
    #              check_lst, lim=total_plot_size, angle_bins=angle_bins)

    # elif method_name == 'no post processing and time and dis filter':
    method_name = option_check[5]
    check_lst = check_time_steps_and_estimate_dif(t_lst, target_real_lst, target_esti_lst)
    plot_results(num_file, target_real_lst[:len(check_lst)], target_esti_lst[:len(check_lst)],
                 rover_lst, x0, y0, map_mean_point, is_exits_real_target,
                 angle_lst[:len(check_lst)], t_lst, snr_lst, method_name, check_lst,
                 lim=total_plot_size, angle_bins=angle_bins)

    print("finish post processing")


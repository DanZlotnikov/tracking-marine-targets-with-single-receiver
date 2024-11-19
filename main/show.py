import os
import shutil
import pickle
import numpy as np
import matplotlib.pyplot as plt

from parameters import args
from numpy.linalg import norm
from utils.plots_utils import plot_calc, clean_path, mp4_ffmpeg
from utils.main_utils import Read_lines, tdoa, rover_utm_location, \
    add_rover_location, fish_plot, create_plot_and_map, take_real_point_from_target, \
    resize_plot, next_step, create_plot_and_map_final, convert_plot_param


def delete_and_create_folders(basic_path, start_path, start_path_rover_img):
    # check if test folder exits
    if not os.path.isdir(basic_path):
        os.mkdir(basic_path)

    # check if experiment is exits.
    if os.path.isdir(start_path):
        # If yes, delete it and create new one.
        shutil.rmtree(start_path)
    os.mkdir(start_path)
    os.mkdir(start_path_rover_img)


def calculate_angle(a, b, c):
    ab, ac, bc = norm(a - b), norm(a - c), norm(b - c)
    alpha = np.arccos((ab ** 2 + bc ** 2 - ac ** 2) / (2 * ab * bc))
    return alpha


def calculate_angle2(a, b, c):
    from target_position_functions.calculation_tdoa_classic import Triangular_rotation
    ab, ac, bc = norm(a - b), norm(a - c), norm(b - c)
    alpha = np.arccos((ab ** 2 + bc ** 2 - ac ** 2) / (2 * ab * bc))

    new_a, new_b, new_c = Triangular_rotation(np.array([a, b, c]))[0]
    siman = (new_a[1] >= 0) * 1 + (new_a[1] < 0) * -1

    return alpha * siman


def calculate_square(rover_lst):
    # rover_lst = np.flip(np.copy(rover_lst), axis=0)
    alpha1 = calculate_angle(rover_lst[0], rover_lst[1], rover_lst[2]) * 180 / np.pi
    alpha2 = calculate_angle(rover_lst[1], rover_lst[2], rover_lst[3]) * 180 / np.pi
    alpha3 = calculate_angle(rover_lst[2], rover_lst[3], rover_lst[0]) * 180 / np.pi
    alpha4 = calculate_angle(rover_lst[3], rover_lst[0], rover_lst[1]) * 180 / np.pi

    check_lst_angle = [True if (val < 150) and (val > 30) else False
                       for val in [alpha1, alpha2, alpha3, alpha4]]

    ab = norm(rover_lst[0] - rover_lst[1])
    bc = norm(rover_lst[1] - rover_lst[2])
    cd = norm(rover_lst[2] - rover_lst[3])
    dc = norm(rover_lst[3] - rover_lst[0])

    check_lst_dis = [True if val > 5 else False
                     for val in [ab, bc, cd, ab]]

    if False in check_lst_angle:
        return False

    return True


if __name__ == '__main__':

    basic_path = 'plots/tests'
    start_path = 'plots/tests/test_' + str(args.num_file)
    start_path_rover_img = start_path + '/rover_path_images'
    delete_and_create_folders(basic_path, start_path, start_path_rover_img)

    # We download the map or use the local maps files and set initial settings in the graph view.
    fig, ax_calc, ax_map, maps, scale_bar = create_plot_and_map(args, show_calc_plot=args.show_calc_plot)
    plt.pause(0.01)
    plt.savefig(start_path_rover_img + '/img_plot1000.png', dpi=150)

    # We're loading the rover's data reporting file
    file_path = os.path.join("data", "test" + str(args.num_file) + '.txt')
    lines = Read_lines(file_path)

    target_idx = -1
    utm_x, utm_y = 0, 0
    idx, count, num_hyper = 0, 0, 0
    fish_pos = np.array([])
    x0, y0, msg_raw = None, None, None
    rover_lst, fish_lst, t_lst, global_time_lst, target_real_idx, snr_lst = [], [], [], [], [], []
    check_fish_plot, is_new_path = False, True
    idx_in_use_lst = []
    full_rover_path_utm = []
    map_mean_point = None
    error_calc_target_lst, target_estimate_loc_rpi = [], []
    angle_lst = []
    count_shift = 0
    check_poly_lst = []
    real_target_lst = []

    # If there is a file that holds details about the location of the target,
    # we will have the option to choose whether to use it or not
    target_real_loc_lst, is_exits_real_target, n_back_look, target_real_global_time_lst = \
        take_real_point_from_target(args, maps)

    # During the experiment, we produce into the file two types of inputs.
    # The location of the rover and the time of arrival of the signal.
    # Each time we go through all the lines in the file and analyze the information.
    for idx, line in enumerate(lines):
        splitLine = line.split(sep=",")

        # Irrelevant lines we ignore.
        if len(splitLine) <= 2:
            pass

        elif splitLine[0] == '#plot_param#':
            map_mean_point = convert_plot_param(params=splitLine)
            maps.mean_point = map_mean_point

        # Lines that hold information about the position of the rover.
        elif splitLine[0] == '#rover_location#':
            # we convert the coordinates from lat&lon to UTM and plot it.
            utm_x, utm_y = rover_utm_location(params=splitLine,
                                              maps=maps,
                                              ax=ax_map,
                                              is_new_path=is_new_path,
                                              plot_lim=args.lim_plot)
            is_new_path = False
            full_rover_path_utm.append([utm_x, utm_y, 0])

        # Lines that hold information about the time arrival.
        elif splitLine[0] == '#target_echo#':
            # We convert our coordinates to the base at the beginning of the axes
            # for ease of calculation according to the rover's first position.
            if x0 is None:
                x0, y0 = utm_x, utm_y

            # we save the last rover location and the time and calculate the target location.
            rover_lst, t_lst, global_time_lst, target_estimate_loc, target_depth, is_shift, snr = \
                add_rover_location(splitLine, utm_x, utm_y,
                                   x0, y0, rover_lst, t_lst,
                                   global_time_lst, maps,
                                   ax_map,
                                   plot_lim=args.lim_plot)

            snr_lst.append(snr)

            # Calculate the position of the object.
            fish_pos, check_fish_plot, check_plot, num_hyper = tdoa(rover_locations=rover_lst,
                                                                    time_lst=t_lst,
                                                                    arguments=args,
                                                                    idx=idx,
                                                                    start_path=start_path,
                                                                    target_depth=target_depth)

            if len(rover_lst) >= int(args.shape) and (fish_pos.size > 0):
                check_poly = calculate_square(np.array(rover_lst[-4:])[:, 0:2])
                check_poly_lst.append(check_poly)

            target_real_point = None
            if is_exits_real_target:
                if target_real_loc_lst.shape[0] > 1:
                    target_idx += 1
                target_real_point = target_real_loc_lst[target_idx, 0:2]
                real_target_lst.append(target_real_point - np.array([x0, y0]))

            fish_pos_new = None
            # Save and plot target position
            if check_fish_plot:
                target_estimate_loc_rpi.append(target_estimate_loc[0])
                error_calc_target = np.linalg.norm(target_estimate_loc[0, 0:2] - fish_pos[0, 0:2])
                error_calc_target_lst.append(error_calc_target)

                if args.show_rpi_calc:
                    fish_pos_new = target_estimate_loc
                else:
                    fish_pos_new = fish_pos

                fish_pos_new, fish_lst = fish_plot(fish_pos_new, fish_lst, maps, x0, y0, ax_map)
                next_location = next_step(rover_lst[-2:], fish_pos_new, args)
                check_next_algo = np.concatenate((np.array(rover_lst[-3:])[:, 0:2], np.array([next_location[0][0:2]])))
                # fig33, ax33 = plt.subplots()
                # ax33.plot(check_next_algo[:, 0], check_next_algo[:, 1])

            # Plot the tdoa calculation
            if check_plot and (ax_calc is not None):
                ax_calc.clear()

                # Plotting on the map requires resizing and sliding, so while we plot the calculation details,
                # we will adjust the plotting to be similar to the map's proportions.
                x_lim, y_lim, zero_axis, rover_lst_plot, fish_pos_plot = resize_plot(maps, rover_lst, fish_pos_new,
                                                                                     args.lim_plot, x0, y0)

                # We plot the real location of the target on the calculation results (if the information exists)
                real_target_plot = None
                if is_exits_real_target:
                    # target_real_loc = np.array([target_real_loc_lst[best_time_idx]])
                    # target_real_loc_plot = maps.convert_point(target_real_loc)
                    real_target_plot = target_real_point - np.array([x0 + zero_axis[0], y0 + zero_axis[1]])

                # Displays the results of the calculation
                plot_calc(ax_calc, fig, rover_lst_plot, t_lst, fish_pos_plot, num_hyper, x_lim=x_lim, y_lim=y_lim,
                          target=real_target_plot)

            # Plot real target location
            if is_exits_real_target:
                maps.plot_real_target(ax_map, np.array([target_real_loc_lst[target_idx, 0:2]]))
                # ax_map.set_box_aspect(1)

                # TODO: remove this part after fixing the bug in is_shift param in ROS branch
                # if is_shift:
                #     count_shift = 1
                # elif count_shift != 0:
                #     count_shift += 1
                #
                # if count_shift >= int(args.shape) and (not is_shift):
                #     count_shift = 0

                if (fish_pos_new is not None) and (count_shift == 0):
                    alpha = calculate_angle2(a=np.array(fish_pos_new)[0:2],
                                             b=np.mean(np.array(rover_lst[-4:])[:, 0:2], axis=0),
                                             c=np.array(target_real_loc_lst[target_idx, 0:2]) - np.array([x0, y0]))
                    angle_lst.append(alpha * 180 / np.pi)

            # save the plots as image and restart the figure
            ax_map.legend().set_visible(args.set_legends_map)
            if ax_calc is not None:
                ax_calc.legend().set_visible(args.set_legends_calc)
            plt.pause(0.01)
            count += 1
            plt.savefig(start_path_rover_img + '/img_plot' + str(1000 + idx) + '.png', dpi=150)
            clean_path(ax_map, maps, rover_lst, fish_lst, x0, y0, args, scale_bar)
            is_clean_old_path = False
            is_new_path = True
            idx_in_use_lst.append(idx)

    # Save all the locations and calculations in the program.
    plt.close('all')
    with open(start_path + '/data_output' + str(args.num_file) + '.pkl', 'wb') as run_data_output:
        pickle.dump([rover_lst, t_lst, fish_lst, idx_in_use_lst, x0, y0, target_real_loc_lst, target_estimate_loc_rpi,
                     check_poly_lst, map_mean_point, is_exits_real_target, angle_lst, snr_lst], run_data_output, 1)

    # Converts a series of images into a video.
    # How to Install h.264 decoder on Ubuntu : https://linuxhint.com/install-h264-decoder-ubuntu/
    mp4_ffmpeg(srt_input=start_path_rover_img + '/img_plot*.png',
               str_output=start_path + '/movie.mp4',
               framerate=0.5,
               isHD=False if ax_calc is None else True)

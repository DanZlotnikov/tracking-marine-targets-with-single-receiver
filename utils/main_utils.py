import os
import shutil

import numpy as np
from numpy.linalg import norm

import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar

from utils.map_utils import create_map

from target_position_functions.calculation_tdoa_classic import TDOA_3points
from target_position_functions.calculation_tdoa_heatmap import tdoa_heat_map
from target_position_functions.calculation_tdoa_optimization import tdoa_optimization
from target_position_functions.calculation_distance_between_hyperbolas_heatmap import heat_map_hyperbola_distance

from rover_position_functions.next_location_calculation import next_location
from rover_position_functions.next_location_triangle_calculation import next_location_triangle


def Read_lines(name):
    file1 = open(name, 'r')
    return file1.readlines()


# We calculate the rover's next steps given his and target's recent locations.
def next_step(rover_locations, fish_location, arguments, n_steps=1, ax=None, fig_plt=None, radius=None):
    n_rover = len(rover_locations)
    rover_locations = np.array(rover_locations)[n_rover - 2:n_rover]
    last_fish_location = fish_location
    if arguments.shape == "triangle_only":
        rover_next_location = next_location_triangle(rov_lst=rover_locations[:, 0:2],
                                                     fish_loc=last_fish_location[0:2],
                                                     max_distance=arguments.max_distance,
                                                     n_steps=n_steps,
                                                     return_2D=False)

    else:
        if radius is None:
            radius = arguments.radius_distance
        rover_next_location = next_location(rov_lst=np.array(rover_locations),
                                            fish_loc=last_fish_location,
                                            radius_length=radius,
                                            n_polygon=int(arguments.shape),
                                            n_steps=n_steps)
    return rover_next_location


# We calculate the target's location given the locations of the rover and the inputs it received from the target.
# By type of calculation, we will take different size of time series from the data.
def tdoa(rover_locations, time_lst, arguments, idx=None, start_path=None, target_depth=0):
    fish_location = np.array([[]])
    is_plot_fish, check_plot = False, False
    num_hyperbola = 0
    if arguments.tdoa_type == "classic_tdoa":
        if len(rover_locations) >= 3:
            xyArr_len = len(rover_locations)
            last_locations = np.array(rover_locations)[xyArr_len - 3:xyArr_len, 0:2]  # 3 points in 2D
            last_times = time_lst[xyArr_len - 3:xyArr_len]
            # last_times = time_mod10(last_times)
            fish_location = TDOA_3points(rov_lst=last_locations,
                                         t_lst=last_times,
                                         velocity=arguments.velocity,
                                         num_file=arguments.num_file,
                                         is_plot=False,
                                         depth=None,
                                         is_3d=False)
            num_hyperbola = 2
            check_plot = True

    elif arguments.tdoa_type == "tdoa_pairs":
        local_shape = 3 if arguments.shape == "triangle_only" else int(arguments.shape)
        if len(rover_locations) >= int(local_shape):
            fish_location_lst = []
            xyArr_len = len(rover_locations)

            last_locations = rover_locations[xyArr_len - local_shape:xyArr_len]
            last_locations = np.array(last_locations)
            last_times = time_lst[xyArr_len - local_shape:xyArr_len]

            # calculate dtoa for each pair of hyperbolas
            count_idx = 0.1
            center_point = np.mean(last_locations[0:4], axis=0)
            for i in np.arange(start=local_shape - 3, stop=-1, step=-1):
                fish_location = TDOA_3points(rov_lst=last_locations[i:i + 3, 0:2],
                                             t_lst=last_times[i:i + 3],
                                             velocity=arguments.velocity,
                                             num_file=arguments.num_file,
                                             is_plot=False,
                                             depth=target_depth,
                                             is_3d=arguments.is_3d)

                # if we manged to calculate the target's location, we add it to the list,
                # otherwise, we calculate the location of the target using heatmaps
                if fish_location.size > 0:
                    for j in range(fish_location.shape[0]):
                        if np.linalg.norm(fish_location[j] - center_point) < 40:
                            fish_location_lst.append(fish_location[j])
                else:

                    new_idx = None
                    if idx is not None:
                        new_idx = idx+count_idx

                    fish_location_heat, isFail = heat_map_hyperbola_distance(rov_lst=last_locations[i:i + 3],
                                                                             p_t=last_times[i:i + 3],
                                                                             size_heat_map=40,
                                                                             size_divide=4,
                                                                             idx=new_idx,
                                                                             start_path=start_path,
                                                                             depth=target_depth,
                                                                             is_3d=arguments.is_3d)
                    fish_location_lst.append(fish_location_heat[0])
                    count_idx += 0.1

            # return fish_location, is_plot_fish, check_plot, num_hyperbola
            check_plot = True
            num_hyperbola = local_shape - 1
            if len(fish_location_lst) == 0:
                check_plot = False
                fish_location = np.array([])
            elif len(fish_location_lst) == 1:
                fish_location = np.array(fish_location_lst)
            else:
                fish_location = np.mean(np.array(fish_location_lst), axis=0)
                fish_location = np.array([fish_location])

    elif arguments.tdoa_type == "optimization_tdoa":
        local_shape = 3 if arguments.shape == "triangle_only" else int(arguments.shape)
        if len(rover_locations) >= int(local_shape):
            xyArr_len = len(rover_locations)
            last_locations = np.array(rover_locations)[xyArr_len - int(local_shape):xyArr_len]
            last_times = time_lst[xyArr_len - int(local_shape):xyArr_len]
            # last_times = time_mod10(last_times)

            last_locations, last_times, mean_point, run_tdoa = check_hyperbola(last_locations, last_times)

            if run_tdoa:
                fish_location, isFail = tdoa_optimization(rov_lst=last_locations,
                                                          t_lst=last_times,
                                                          velocity=arguments.velocity,
                                                          p_t=mean_point,
                                                          is_2d=False)
                fish_location[0, 2] = target_depth

            else:
                fish_location, isFail = np.array([]), True

            num_hyperbola = len(last_locations) - 1
            check_plot = True

    elif arguments.tdoa_type == "heat_map_tdoa":
        if len(rover_locations) >= int(arguments.shape):
            xyArr_len = len(rover_locations)
            last_locations = np.array(rover_locations)[xyArr_len - int(arguments.shape):xyArr_len]
            last_times = time_lst[xyArr_len - int(arguments.shape):xyArr_len]
            # last_times = time_mod10(last_times)

            last_locations, last_times, _, run_tdoa = check_hyperbola(last_locations, last_times)
            if run_tdoa:
                fish_location, isFail = tdoa_heat_map(rov_lst=last_locations,
                                                      p_t=last_times,
                                                      idx=idx,
                                                      size_heat_map=40,
                                                      size_divide=4,
                                                      start_path=start_path)
                fish_location[0, 2] = target_depth
            else:
                fish_location, isFail = np.array([]), True

            num_hyperbola = len(last_locations) - 1
            check_plot = True

    elif arguments.tdoa_type == "heat_map_distance":
        if len(rover_locations) >= int(arguments.shape):
            xyArr_len = len(rover_locations)
            last_locations = np.array(rover_locations)[xyArr_len - int(arguments.shape):xyArr_len]
            last_times = time_lst[xyArr_len - int(arguments.shape):xyArr_len]
            # last_times = time_mod10(last_times)

            fish_location, isFail = heat_map_hyperbola_distance(rov_lst=last_locations,
                                                                p_t=last_times,
                                                                size_heat_map=40,
                                                                size_divide=4,
                                                                idx=idx,
                                                                start_path=start_path,
                                                                depth=target_depth,
                                                                is_3d=arguments.is_3d)
            fish_location[0, 2] = target_depth
            num_hyperbola = len(last_locations) - 1
            check_plot = True

    if fish_location.size > 0:
        is_plot_fish = True

    return fish_location, is_plot_fish, check_plot, num_hyperbola


def check_hyperbola(last_locations, last_times):
    n = len(last_locations)
    from utils.time_utils import calculate_time_difference
    t_lst = calculate_time_difference(last_times)
    rd_lst = t_lst * 1503
    # rd_lst = 1503 * (last_times[0:n - 1] - last_times[1:n])
    canonical_hyperbola_b_value = (norm(last_locations[0:n - 1] - last_locations[1:n], axis=1) / 2) ** 2 - (
            abs(rd_lst) / 2) ** 2
    is_hyperbola = (canonical_hyperbola_b_value > 0)
    NRD_last = rd_lst[-1] / norm(last_locations[-1] - last_locations[-2])

    if np.sum(is_hyperbola) == is_hyperbola.size:
        run_tdoa = True
    elif np.sum(is_hyperbola[-2:]) == 2:
        run_tdoa = True
        valid_data_idx = np.max(np.where(is_hyperbola == False)) + 1
        last_locations = last_locations[valid_data_idx:]
        last_times = last_times[valid_data_idx:]
        print("we use only part from the data")
    else:
        run_tdoa = False
        print("ERROR: there exits non hyperbola")

    if NRD_last > 0.7:
        mean_point = last_locations[-1] + np.random.random(1)
    elif NRD_last < -0.7:
        mean_point = last_locations[-2] + np.random.random(1)
    else:
        mean_point = np.mean(last_locations, 0) - np.random.random(1)

    return last_locations, last_times, mean_point, run_tdoa


def check_nrd(last_locations, last_times):
    NRD_lst = []
    n = len(last_locations)
    rd_lst = 1503 * (last_times[0:n - 1] - last_times[1:n])
    for i in np.arange(start=-1, stop=-n, step=-1):
        NRD_lst.append(rd_lst[i] / norm(last_locations[i] - last_locations[i - 1]))
    return NRD_lst


# Delete graphs of previous experiments
def delete_old_files():
    if os.path.isdir("plots/temp_images"):
        shutil.rmtree("plots/temp_images")
    os.mkdir("plots/temp_images")

    if os.path.exists("plots/movie.mp4"):
        os.remove("plots/movie.mp4")


# Given the input from the file, we will convert it to UTM coordinates and display them on the map.
def rover_utm_location(params, maps, ax, is_new_path, plot_lim=None):
    utm_x, utm_y, _, global_time = convert_coordinate_utm(params, str_type='rover_location')
    point_plot = maps.plot_point(np.array([[utm_x, utm_y]]), ax, '*', s=50, c='w', plot_lim=plot_lim, width=0.1)
    if is_new_path:
        point_plot.set_label("Rover's route from the last point")
        ax.legend()
        # ax.set_box_aspect(1)
    return utm_x, utm_y


def convert_coordinate_utm(params, str_type=None):
    if str_type == "real_target_location":
        global_time = int(params[1][params[1].find(":") + 1:])
        real_target_location_utm = np.array([float(params[6][params[6].find(":") + 1:]),
                                             float(params[7][params[7].find(":") + 1:])])
        depth = float(params[8][params[8].find(":") + 1:])
        return real_target_location_utm[0], real_target_location_utm[1], depth, global_time
    elif str_type == 'rover_location':
        global_time = int(params[1][params[1].find(":") + 1:])
        real_target_location_utm = np.array([float(params[2][params[2].find(":") + 1:]),
                                             float(params[3][params[3].find(":") + 1:])])
        depth = 0
        return real_target_location_utm[0], real_target_location_utm[1], depth, global_time


def convert_plot_param(params):
    map_mean_point = np.array([[float(params[1][params[1].find(":") + 1:]),
                                float(params[2][params[2].find(":") + 1:]), 0]])
    return map_mean_point


# Each time we receive input from the tag, we will keep the location of the rover and the signal arrival time.
def add_rover_location(split_line, utm_x, utm_y, x0, y0, rover_lst, t_list, global_time_lst, maps, ax, plot_lim=50, min_dist=0.1):
    global_time = int(split_line[1][split_line[1].find(":") + 1:])
    global_time_lst.append(global_time)
    t_sec = float(split_line[2][split_line[2].find(":") + 1:])
    t_mil = float(split_line[3][split_line[3].find(":") + 1:])
    t = t_sec + t_mil / 1000.0

    target_depth = float(split_line[8][split_line[8].find(":") + 1:])
    # t = t - target_depth * 1503

    point_plot = maps.plot_point(np.array([[utm_x, utm_y]]), ax, None,
                                 s=150, width=None, plot_lim=plot_lim, c='royalblue')
    point_plot.set_label("Rover last location")
    ax.legend()
    # ax.set_box_aspect(1)

    target_estimate_loc_utm = np.array([[float(split_line[4][split_line[4].find(":") + 1:]),
                                         float(split_line[5][split_line[5].find(":") + 1:]), 0]])
    if target_estimate_loc_utm[0, 0] == 0.0 and target_estimate_loc_utm[0, 1] == 0.0:
        target_estimate_loc = target_estimate_loc_utm
    else:
        target_estimate_loc = target_estimate_loc_utm - np.array([x0, y0, 0])

    # In order to make it easy to work with the data,
    # we convert the base coordinates to the beginning of the axes
    utm_x -= x0
    utm_y -= y0

    dist = 0
    if not rover_lst == []:
        dist = norm(np.array([utm_x, utm_y, 0]) - np.array(rover_lst[-1]))

    # We require a minimum distance between each point where the Rover has received a signal.
    if (rover_lst == []) | (dist >= min_dist):  # args.min_distance):
        rover_lst.append([utm_x, utm_y, 0])
        t_list.append(t)

    is_shift = int(float(split_line[9][split_line[9].find(":") + 1:])) == 1
    snr = int(float(split_line[10][split_line[10].find(":") + 1:]))

    return rover_lst, t_list, global_time_lst, target_estimate_loc, target_depth, is_shift, snr


# If we were able to calculate the target's location, we would represent it on the map and graph.
# TODO : For now, if we get more than one solution, we will always choose the first of them.
def fish_plot(fish_position, fish_position_lst, maps, x0, y0, ax):
    if fish_position.size > 0:
        fish_position_lst.append(fish_position[0])
        point_plot = maps.plot_point(np.array([fish_position[0, 0:2]]) + np.array([x0, y0]),
                                     ax, None, s=150, c='orange', set_xy_lim=False, width=None)
        point_plot.set_label("Estimated target location")
        ax.legend()
        # ax.set_box_aspect(1)
        return fish_position[0], fish_position_lst
    else:
        return np.array([[]]), fish_position_lst


# We download the map or use the local maps files and set initial settings in the graph view.
def create_plot_and_map(args, show_calc_plot=True):
    if show_calc_plot:
        fig, (ax_calc, ax_map) = plt.subplots(1, 2, figsize=(16.0, 16.0))
        ax_calc.set_box_aspect(1)
        ax_map.set_box_aspect(1)

        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())

        ax_calc.xaxis.set_visible(False)
        ax_calc.yaxis.set_visible(False)
    else:
        fig, ax_map = plt.subplots(1, 1, figsize=(16.0, 16.0))
        ax_calc = None

    fig.show()

    lt_lat, lt_lng, rb_lat, rb_lng = np.array(args.map_cod)
    maps = create_map(lt_lat, lt_lng, rb_lat, rb_lng, args.map_name, style=args.map_style)
    scale_bar = ScaleBar(maps.pixel_length, location="lower right")  # 1 pixel = 0.2 meter
    maps.reset_plot(ax_map, scale_bar)
    plt.pause(1)

    # delete_old_files()

    return fig, ax_calc, ax_map, maps, scale_bar


def create_plot_and_map_final(args, map_mean_point=None):
    fig, ax_map = plt.subplots(1, 1, figsize=(16.0, 16.0))
    ax_map.set_box_aspect(1)
    mng = plt.get_current_fig_manager()
    # mng.resize(*mng.window.maxsize())
    fig.show()

    lt_lat, lt_lng, rb_lat, rb_lng = np.array(args.map_cod)
    maps = create_map(lt_lat, lt_lng, rb_lat, rb_lng, args.map_name, style=args.map_style)
    if map_mean_point is not None:
        maps.mean_point = map_mean_point[0:2]
    scale_bar = ScaleBar(maps.pixel_length, location="lower right")  # 1 pixel = 0.2 meter
    maps.reset_plot(ax_map, scale_bar)
    plt.pause(1)

    return fig, ax_map, maps, scale_bar


# If there is a file that holds details about the location of the target,
# we will have the option to choose whether to view it or not
def take_real_point_from_target(args, maps):
    file_path = os.path.join("data", "test" + str(args.num_file) + '.txt')
    file_type = args.convert_coordinate_utm

    is_static = (args.target_real_location == 'Static')
    is_dynamic = (args.target_real_location == 'Dynamic')

    if not os.path.isfile(file_path) or args.target_real_location == 'None':
        return np.array([[]]), False, 0, None

    lines = Read_lines(file_path)
    target_lst = []
    global_time_lst = []

    for line in lines:
        splitLine = line.split(sep=",")

        # Irrelevant lines we ignore.
        if len(splitLine) <= 2:
            pass

        elif splitLine[0] == '#target_echo#':
            utm_x, utm_y, depth, global_time = convert_coordinate_utm(params=splitLine,
                                                                      str_type="real_target_location")
            target_lst.append([utm_x, utm_y, depth])
            global_time_lst.append(global_time)

    global_time_lst = np.array(global_time_lst)
    if is_static:
        return np.array([target_lst[-1]]), True, 0, None
    elif is_dynamic:
        num_prev = (3 - 2) if args.shape == "triangle_only" else int(args.shape) - 2
        return np.array(target_lst), True, 0, global_time_lst
    else:
        return np.array([[]]), False, None


# Adjusting the calculation plotting to be similar to the map's proportions.
def resize_plot(maps, rover_lst, fish_pos, lim_plot, x0, y0):
    # Calculate rescale of the plot.
    point_on_map = np.array(maps.convert_point(np.array(rover_lst) + np.array([x0, y0, 0]))).T
    distance_on_map = np.linalg.norm(point_on_map[-1] - point_on_map[-2])
    distance_on_calc = np.linalg.norm(np.array(rover_lst)[-1] - np.array(rover_lst)[-2])
    resize_value = distance_on_calc / distance_on_map

    if maps.mean_point is not None:
        # Move all the points to the beginning of the axes in relation to pre define map's mean point.
        zero_axis = maps.mean_point[0] - np.array([x0, y0, 0])
    else:
        # Move all the points to the beginning of the axes in relation to the last position of the Rover.
        zero_axis = np.array(rover_lst)[-1]

    rover_lst_plot = np.array(rover_lst) - zero_axis
    fish_pos_plot = fish_pos
    if len(fish_pos) > 0:
        fish_pos_plot = np.array(fish_pos) - zero_axis
    x_lim = np.array([rover_lst_plot[-1][0] - lim_plot, rover_lst_plot[-1][0] + lim_plot]) * resize_value
    y_lim = np.array([rover_lst_plot[-1][1] - lim_plot, rover_lst_plot[-1][1] + lim_plot]) * resize_value

    return x_lim, y_lim, zero_axis, rover_lst_plot, fish_pos_plot

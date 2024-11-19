from __future__ import print_function

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

import shutil
import numpy as np
import pandas as pd
import tqdm
import matplotlib.pyplot as plt
from utm import from_latlon, to_latlon
from matplotlib_scalebar.scalebar import ScaleBar

from parameters import args
from utils.time_utils import get_time
from utils.main_utils import tdoa, next_step
from utils.simulator_utils import rand_path, delete_old_files, error_fn, check_polygon_length, plot_target_path
from utils.simulator_utils import plot_simulator_with_receiver, plot_complete_simulator, plot_animator_simulator
from utils.plots_utils import distance_error_plot, distance_error_no_plot

import gc
import pickle

project_path = args.current_path

def run_single_simulator(target_location_lst, map_loc, start_path, vehicle, args, proj=None, sitl=None):
    num_hyper = 0
    travel_report_lst = np.array([True, True])
    interception_lst = np.ones(len(target_location_lst))

    start_inter = 20
    interception_lst[start_inter:] = np.random.choice(2, len(target_location_lst) - start_inter, p=[0.5, 0.5])
    interception_lst = [k == 1 for k in interception_lst]

    if args.always_get_update_on_calculation_error:
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())

    target_location_lst = target_location_lst + map_loc
    target_calc_lst = np.array(target_location_lst)[0:2]
    rover_location_lst = np.array(args.rover_start_location) + map_loc
    time_lst = np.array([])

    # Calculate signal arrival time from target to rover with predefined time accuracy.
    t = get_time(rover_location_lst[0], target_location_lst[0], args.velocity, 0, args.time_accuracy)
    time_lst = np.append(time_lst, t)

    t = get_time(rover_location_lst[1], target_location_lst[1], args.velocity, 10, args.time_accuracy)
    time_lst = np.append(time_lst, t)

    for i in tqdm.tqdm(range(args.maxIter - 2)):
        if (args.use_interception and interception_lst[i + 2]) or not args.use_interception:
            # We calculate the rover's next steps given his and target's recent locations.
            next_location_error = error_fn(sigma=args.next_loc_delta, size=3)
            next_location = next_step(rover_location_lst, target_calc_lst[i + 1], args, radius=args.radius_distance)
            if next_location.shape[1] <= 2:
                next_location = np.array([[next_location[0, 0], next_location[0, 1], 0]])
            next_location += next_location_error
            is_arrived_destination = True
            current_location = next_location[0]
            rover_location_lst = np.append(rover_location_lst, np.array([current_location]), axis=0)
            travel_report_lst = np.append(travel_report_lst, is_arrived_destination)

            # Calculate signal arrival time from target to rover with predefined time accuracy.
            t_error = error_fn(sigma=args.t_delta, size=1)
            t = get_time(rover_location_lst[-1],
                         target_location_lst[i + 2],
                         args.velocity, 10 * (i + 2),
                         args.time_accuracy)
            t += t_error
            time_lst = np.append(time_lst, t)

            # add an error to the rover's location
            location_error = error_fn(sigma=args.loc_delta, size=3)
            rover_location_lst[-1] += location_error

            # Calculate the position of the object.
            target_calc_loc, check_fish_plot, check_plot, num_hyper = tdoa(rover_location_lst, time_lst, args)

            # If several solutions are obtained, we will always use one of the solutions.
            # If no solution is obtained, we will use the previous location
            if check_fish_plot:
                target_calc_loc = target_calc_loc[0]
            else:
                target_calc_loc = target_calc_lst[-1]
            target_calc_lst = np.append(target_calc_lst, np.array([target_calc_loc]), axis=0)

            if num_hyper >= 2 and args.always_get_update_on_calculation_error:
                plt.close("all")
                distance_error_plot(target_calc_lst, target_location_lst, travel_report_lst, i=0)
                plt.savefig(start_path + '/distance_error_sim_' + args.sim_name + '.png', dpi=150)
        else:
            target_calc_loc = target_calc_lst[-1]
            target_calc_lst = np.append(target_calc_lst, np.array([target_calc_loc]), axis=0)

    plt.close("all")
    # compare the results
    if args.use_dronekit_sitl:
        print("The number of times the Rover has reached its destination: " + str(np.sum(travel_report_lst))
              + "/" + str(travel_report_lst.shape[0]))

    if args.use_dronekit_sitl and args.use_interception:
        check_lst = [(k and z) for k, z in zip(travel_report_lst, interception_lst)]
    elif args.use_dronekit_sitl:
        check_lst = travel_report_lst
    elif args.use_interception:
        check_lst = interception_lst
    else:
        check_lst = [True] * len(target_location_lst)

    # mng = plt.get_current_fig_manager()
    # mng.resize(*mng.window.maxsize())
    # distance_error_plot(target_calc_lst, target_location_lst, check_lst, i=0, n_cut=0)
    # plt.savefig(start_path + '/distance_error_sim_' + args.sim_name + '.png', dpi=150)

    # Saving simulation data:
    xy_error, xy_error_mean, xy_error_rms, xy_error_median, xy_error_std = \
        distance_error_no_plot(target_calc_lst, target_location_lst, check_lst, i=0, n_cut=0)

    sim_data = pd.DataFrame({'rover location': list(rover_location_lst - map_loc),
                             'real target location': list(target_location_lst[check_lst] - map_loc),
                             'estimate target location': list(target_calc_lst[check_lst] - map_loc),
                             'travel report': [True] * len(rover_location_lst),
                             'time': time_lst,
                             'num_hyper': np.ones(len(time_lst)) * num_hyper})
    sim_data.to_csv(start_path + '_mean_error_' + str(np.around(xy_error_mean, decimals=1)) + '.csv')

    return xy_error, xy_error_mean, xy_error_rms, xy_error_median, xy_error_std


def run_simulation(args, target_speed, new_targer_road, map_loc, radius_lst, n_polugon_size_lst, vehicle):
    df, df_summery = pd.DataFrame(), pd.DataFrame()
    sim_option = "sim2" if args.use_interception else "sim"

    # This function creates a random trajectory for the target with preferred directionality.
    if new_targer_road:
        target_location_lst = rand_path(start_location=np.array(args.target_start_location),
                                        maxIt=args.maxIter - 3,
                                        length=target_speed)
        target_location_df = pd.DataFrame(target_location_lst)
        target_location_df.to_csv(args.sim_path + f'/plots/{sim_option}/target_path.csv')
    else:
        target_location_df_temp = pd.read_csv(args.sim_path + f'/plots/{sim_option}/target_path.csv')
        target_location_lst = target_location_df_temp.to_numpy()[:, 1:]

    # plot_target_path(target_location_lst, args.sim_path + f'/plots/{sim_option}/target_path.png')
    # plt.close("all")

    for check_n_polugon in n_polugon_size_lst:
        for check_radius in radius_lst:

            n_test = 3
            xy_error_mean_lst = np.zeros(n_test)
            xy_error_rms_lst = np.zeros(n_test)
            xy_error_median_lst = np.zeros(n_test)
            xy_error_std_lst = np.zeros(n_test)

            polygon_dist = int(np.around(check_polygon_length(check_n_polugon, check_radius), decimals=0))
            start_path = (args.sim_path + f'/plots/{sim_option}/test_shape_' +
                          str(check_n_polugon) + '_length_' + str(polygon_dist))

            args.max_distance = polygon_dist
            args.radius_distance = check_radius
            args.shape = str(check_n_polugon)  # 'triangle_only'

            for s in range(n_test):
                xy_error, xy_error_mean, xy_error_rms, xy_error_median, xy_error_std = \
                    run_single_simulator(target_location_lst, map_loc,
                                         start_path, vehicle, args, proj=None, sitl=None)

                xy_error_mean_lst[s] = xy_error_mean
                xy_error_rms_lst[s] = xy_error_rms
                xy_error_median_lst[s] = xy_error_median
                xy_error_std_lst[s] = xy_error_std

                df = pd.concat([df, pd.DataFrame.from_dict({'shape': check_n_polugon,
                                                            'radius': check_radius,
                                                            'length': polygon_dist,
                                                            'mean': xy_error_mean,
                                                            'rms': xy_error_rms,
                                                            'median': xy_error_median,
                                                            'std': xy_error_std}, orient='index').T])

            df_summery = pd.concat([df_summery, pd.DataFrame.from_dict({'shape': check_n_polugon,
                                                                        'radius': check_radius,
                                                                        'length': polygon_dist,
                                                                        'mean': np.mean(xy_error_mean_lst),
                                                                        'mean_std': np.std(xy_error_mean_lst),
                                                                        'rms': np.mean(xy_error_rms_lst),
                                                                        'rms_srtd': np.std(xy_error_rms_lst),
                                                                        'median': np.mean(xy_error_median_lst),
                                                                        'median_std': np.std(xy_error_median_lst),
                                                                        'std': np.mean(xy_error_std_lst),
                                                                        'std_std': np.std(xy_error_std_lst)},
                                                                       orient='index').T])

    df.to_csv(args.sim_path + f'/plots/{sim_option}/results_' + args.tdoa_type + '.csv')
    df_summery.to_csv(args.sim_path + f'/plots/{sim_option}/results_' + args.tdoa_type + '_summery.csv')


def plot_simulation(file_name: str):
    sim_option = "sim2" if args.use_interception else "sim"
    full_path = f'{project_path}/plots/{sim_option}/'
    df = pd.read_csv(full_path + file_name)
    rover_location_lst = np.array([np.fromstring(val.replace("[", "").replace("]", ""), dtype=float, sep=' ')
                                   for val in df['rover location'].values])
    target_real_location_lst = np.array([np.fromstring(val.replace("[", "").replace("]", ""), dtype=float, sep=' ')
                                         for val in df['real target location'].values])
    target_esti_location_lst = np.array([np.fromstring(val.replace("[", "").replace("]", ""), dtype=float, sep=' ')
                                         for val in df['estimate target location'].values])
    time_lst = df['time'].values
    num_hyper = df['num_hyper'].values[0]
    travel_report_lst = df['travel report'].values

    # distance_error_plot(target_esti_location_lst, target_real_location_lst, travel_report_lst, i=0, n_cut=0)
    # plt.savefig(full_path + args.sim_name + '_error_plot.png', dpi=150)
    # plt.close('all')

    start_path = args.sim_path + f'/temp_images/{sim_option}'

    # plot_animator_simulator(rover_location_lst, target_real_location_lst, target_esti_location_lst, n_steps=4,
    #                         is_end=True)
    # fig, ax = plot_complete_simulator(rover_location_lst, target_real_location_lst, target_esti_location_lst)
    # fig.savefig(full_path + args.sim_name + '_plot_sim.png', dpi=150)
    # plt.close('all')

    # fig, ax = plot_simulator_with_receiver(rover_location_lst, target_real_location_lst, target_esti_location_lst,
    #                              time_lst, args.maxIter, num_hyper, start_path)
    # fig.savefig(full_path + args.sim_name + '_plot_sim_partial.png', dpi=150)
    # plt.close('all')

sim_option = "sim2" if args.use_interception else "sim"
full_path = f'{project_path}/plots/{sim_option}/'
if not os.path.exists(full_path):
    os.makedirs(full_path)

if __name__ == '__main__':
    sitl, vehicle = None, None
    proj = None  # Proj("+proj=utm +zone=55H +south +ellps=WGS84 +datum=WGS84 +units=m +no_defs")

    target_speed = 2.0  # distance per 10 seconds
    new_target_road = True
    radius_lst = [8, 10, 12, 14, 16]
    n_polugon_size_lst = ['triangle_only', 3, 4, 5, 6]
    args.tdoa_type = 'optimization_tdoa'  # 'optimization_tdoa'  # 'tdoa_pairs'

    lat, lon = -35.3632596, 149.1652268
    utm_param = np.array(from_latlon(lat, lon))
    utm_x, utm_y, zone_num, zone_letter = float(utm_param[0]), float(utm_param[1]), int(utm_param[2]), utm_param[3]
    map_loc = np.array([utm_x, utm_y, 0])

    is_run, is_plot = True, False

    if is_run:
        run_simulation(args, target_speed, new_target_road, map_loc, radius_lst, n_polugon_size_lst, vehicle)
    if is_plot:
        # 1. test_shape_3_length_20_mean_error_2.6
        # 2. test_shape_4_length_14_mean_error_3.5
        plot_simulation(file_name='test_shape_4_length_14_mean_error_7.1.csv')

    print("123")

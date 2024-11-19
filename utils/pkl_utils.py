import os
import pickle
import numpy as np
from utm import from_latlon, to_latlon
import pandas as pd
import datetime
from datetime import timedelta

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))


def global_time_to_str(global_time):
    if (global_time != 0) and (global_time != 0.0):
        return str(global_time)
    return '0'


def target_to_str(global_time=0, t=0.0, start_time=0.0, depth=0.0,
                  estimate_location=None, real_location=None, is_shift=0, snr=0):
    if real_location is None:
        location_real = [0, 0]
    if estimate_location is None:
        location_estimate = [0, 0]
    global_time_str = global_time_to_str(global_time)
    t_sec = int(t) + start_time
    t_mil = int((t - int(t)) * 1000)
    echo_str = '#target_echo#' + \
               ',global_time:' + global_time_str + \
               ',receiver_time_sec:' + str(t_sec) + \
               ',receiver_time_mil:' + str(t_mil) + \
               ',estimate_location_utm_x:' + str(estimate_location[0]) + \
               ',estimate_location_utm_y:' + str(estimate_location[1]) + \
               ',real_location_utm_x:' + str(real_location[0]) + \
               ',real_location_utm_y:' + str(real_location[1]) + \
               ',target_depth:' + str(np.abs(depth)) + \
               ',is_shift:' + str(is_shift) + \
               ',snr:' + str(snr) + '\n'

    return echo_str


def location_to_str(global_time=0, rover_utm=None, device_name='rover_location'):
    if rover_utm is None:
        rover_utm = [0, 0]
    global_time_str = global_time_to_str(global_time)
    loca_str = '#' + device_name + '#' + \
               ',global_time:' + global_time_str + \
               ',utm_x:' + str(rover_utm[0]) + \
               ',utm_y:' + str(rover_utm[1]) + '\n'
    return loca_str


def plot_param_to_str(mean_point=None):
    if mean_point is None:
        mean_point = [0, 0]
    plot_str = '#plot_param#' + \
               ',mean_point_x:' + str(mean_point[0]) + \
               ',mean_point_y:' + str(mean_point[1]) + '\n'

    return plot_str


def convert_lines(data_string, is_all_num=True):
    new_line = data_string.replace('\n', '') \
        .replace("\'", '') \
        .replace(",", ' ') \
        .replace("[", ' ') \
        .replace("]", ' ').split()
    if 'dev' in new_line:
        new_line = [new_line[0], new_line[1], new_line[2]]  #[new_line[0], new_line[-2], new_line[-1]]
    if is_all_num:
        new_line = [float(val) for val in new_line]
    return new_line


def convert_pkl(rover_file_num, target_file_num, is_static=False,
                new_depth=False, old_files=False, calculate_only_target_data=False):
    full_path = f'{parent_dir}/data/'

    if old_files:
        with open(full_path + "pkl/" + 'ROS_param' + str(rover_file_num) + '.txt') as f:
            data_str_lst = f.readlines()
    else:
        with open(full_path + "pkl/" + 'ROS_param' + str(rover_file_num) + '_ll.txt') as f:
            data_str_lst = f.readlines()

    with open(full_path + "pkl/" + 'ROS_param' + str(rover_file_num) + '.txt') as f:
        data_str_lst_for_depth = f.readlines()

    if not old_files:
        with open(full_path + "pkl/" + 'logs' + str(rover_file_num) + '.txt') as f:
            data_log = f.readlines()
            data_log_lst = [convert_lines(data_log[i], False) for i in range(len(data_log))]

    data_target_lst = []
    if is_static:
        if old_files:
            with open(full_path + "pkl/" + 'target_location' + str(target_file_num) + '.pkl', 'rb') as f:
                target_lat_lon = pickle.load(f)
        else:
            with open(full_path + "pkl/" + 'target_location' + str(target_file_num) + '.pkl', 'rb') as f:
                _, target_lat_lon, _ = pickle.load(f)
        global_time = [0]
        target_utm = list(from_latlon(target_lat_lon[0], target_lat_lon[1])[0:2])
        data_target_lst.append(global_time + target_utm)
    else:
        with open(full_path + "pkl/" + 'gps_points_file' + str(target_file_num) + '.txt') as f:
            target_real_loc_lst = f.readlines()

        target_real_loc_lst = target_real_loc_lst[:-1]
        for data_str in target_real_loc_lst:
            data = convert_lines(data_str)
            global_time, target_lat, target_lon = data
            global_time = [int(global_time)]
            target_utm = list(from_latlon(target_lat, target_lon)[0:2])
            data_target_lst.append(global_time + target_utm)

    data_target_lst = np.array(data_target_lst)

    # global_time = 0
    str_lst = []
    # next_location_lst = []
    target_location_lst = []
    rover_utm_lst = []
    save_rover_loc = True

    serial_lst, x_lst, y_lst, z_lst, syns_tag, idx_lst = [], [], [], [], [], []
    idx_count, serial_count = 1, 100001

    date_and_time_lst, receiver_name_lst, transmitter_name, time_mod10 = [], [], [], []

    depth_lst = []
    for data_str in data_str_lst_for_depth[:-1]:
        data = convert_lines(data_str)
        if len(data) > 3:
            transmit_value = data[8]
            resolution = 0.1000
            calibrate_absolute_pressure = 1000
            target_depth = transmit_value * resolution - (calibrate_absolute_pressure - 1000) / 100
            depth_lst.append(target_depth)

    count_depth = -1
    global_time_start, global_time_end = None, None
    for data_str in data_str_lst[:-1]:
        data = convert_lines(data_str)

        if len(data) > 3:
            save_rover_loc = True
            # rover_location = data[0:3]
            # x0, y0 = data[3:5]
            count_depth += 1
            t = data[5]
            target_location_estimate = data[6:9]
            target_depth = data[8] if new_depth else depth_lst[count_depth] # data[8]
            next_location = data[9:11]
            global_time = 0
            if len(data) >= 12:
                global_time = int(data[11])

            if global_time_start is None:
                global_time_start = global_time
            global_time_end = global_time

            if len(data) >= 13:
                is_shift = int(data[12])
            else:
                is_shift = 0

            if len(data) >= 14:
                snr = int(data[13])
            else:
                snr = 0

            target_location_lst.append(target_location_estimate)
            idx = np.argmin(np.abs(data_target_lst[:, 0] - global_time))
            str_lst.append(location_to_str(global_time=global_time,
                                           rover_utm=rover_utm,
                                           device_name='rover_location'))
            data_str = target_to_str(global_time=global_time,
                                     t=t,
                                     start_time=0,
                                     estimate_location=target_location_estimate[:2],
                                     real_location=data_target_lst[idx, 1:3],
                                     depth=target_depth,
                                     is_shift=is_shift,
                                     snr=snr)
            str_lst.append(data_str)

            # Hydros table
            serial_lst.append(serial_count)
            idx_lst.append(idx_count)
            syns_tag.append(None)  # 12345
            z_lst.append(0.01)
            x_lst.append(int(rover_utm[0]) + 0.0)
            y_lst.append(int(rover_utm[1]) + 0.0)

            # detection table
            utc_time = timedelta(seconds=t)
            hour, minute, seconds_and_micro = str(utc_time).split(':')
            seconds, microsecond = str(utc_time).split(':')[-1].split('.')
            utc_time_and_date = datetime.datetime(year=2022, month=1, day=1,
                                                  hour=int(hour), minute=int(minute),
                                                  second=int(seconds), microsecond=int(microsecond))
            date_and_time_lst.append(utc_time_and_date)
            receiver_name_lst.append("VR2W-" + str(serial_count))
            transmitter_name.append("A96-1111-11111")
            time_mod10.append(t % 10)

            serial_count += 1
            idx_count += 1

        elif len(data) == 3:
            if True:
                global_time = 0  # global_time = data[0]
                rover_utm = list(from_latlon(data[1], data[2])[0:2])
                rover_utm_lst.append(rover_utm)
                str_lst.append(location_to_str(global_time=global_time,
                                               rover_utm=rover_utm,
                                               device_name='rover_location'))
                save_rover_loc = False

    if data_target_lst.shape[0] > 1:
        target_analytics = data_target_lst[(data_target_lst[:, 0] >= global_time_start) &
                                           (data_target_lst[:, 0] <= global_time_end)]
        target_analytics_dt = (target_analytics[1:] - target_analytics[0:-1])[:, 0]
        target_analytics_dif_dis = (target_analytics[1:] - target_analytics[0:-1])[:, 1:]
        target_analytics_res = np.linalg.norm(target_analytics_dif_dis, axis=1) / target_analytics_dt

        target_analytics_mean = np.mean(target_analytics_res)
        target_analytics_std = np.std(target_analytics_res)
        target_analytics_max = np.max(target_analytics_res)
        target_analytics_min = np.min(target_analytics_res)
        target_analytics_total_dis = np.sum(np.linalg.norm(target_analytics_dif_dis, axis=1))
        target_analytics_total_time = target_analytics[-1, 0] - target_analytics[0, 0]
        target_analytics_total_detections = len(target_location_lst) #

    else:
        target_analytics_mean = 0
        target_analytics_std = 0
        target_analytics_max = 0
        target_analytics_min = 0
        target_analytics_total_dis = 0
        target_analytics_total_time = global_time_end - global_time_start
        target_analytics_total_detections = len(target_location_lst)

    target_analytics_dict = {'mean': target_analytics_mean,
                             'std': target_analytics_std,
                             'max': target_analytics_max,
                             'min': target_analytics_min,
                             'total_dis': target_analytics_total_dis,
                             'total_time': target_analytics_total_time,
                             'total_detections': target_analytics_total_detections}

    target_analytics_df = pd.DataFrame(target_analytics_dict, index=[0])
    target_analytics_df.to_csv(full_path + "debug/" + 'target_analytics_' + str(rover_file_num) + '.csv', index=False)


    if calculate_only_target_data:
        return None

    syns_tag[-1] = 12345
    syns_tag[-2] = 12346
    syns_tag[-3] = 12347
    mean_point1 = np.array([np.mean(x_lst), np.mean(y_lst)])
    mean_point2 = np.array([np.mean(x_lst)-10, np.mean(y_lst)])
    mean_point3 = np.array([np.mean(x_lst), np.mean(y_lst)+10])
    df_hydros = pd.DataFrame({'serial': serial_lst,
                              'x': x_lst,
                              'y': y_lst,
                              'z': z_lst,
                              'sync_tag': syns_tag,
                              'idx': idx_lst})

    new_date_and_time_lst = []
    new_receiver_name_lst = []
    new_transmitter_name = []
    for i in np.arange(start=4, stop=len(date_and_time_lst)):
        last_recivers_id = receiver_name_lst[i - 4:i]
        last_recivers_time = date_and_time_lst[i - 4:i]
        last_transmiter_name = transmitter_name[i - 4:i]
        time_mod10_last = time_mod10[i - 4:i]
        dt_last_lst = np.array(time_mod10_last[1:4]) - np.array(time_mod10_last[0:3])
        positive_lst = (dt_last_lst < 0) * (-1) + (dt_last_lst >= 0) * (1)
        #
        # last_time = last_recivers_time[-1]
        # new_times = [last_time]
        # for j in range(3):
        #     last_time = last_time + positive_lst[j] * timedelta(seconds=np.abs(dt_last_lst[j]))
        #     new_times.append(last_time)

        new_times = []
        for j in range(3):
            dt_time = last_recivers_time[-1] - last_recivers_time[j]
            new_times.append(last_recivers_time[j] + dt_time - timedelta(microseconds=dt_time.microseconds))
        new_times.append(last_recivers_time[-1])


        new_date_and_time_lst += list(new_times)
        new_receiver_name_lst += list(last_recivers_id)
        new_transmitter_name += list(last_transmiter_name)

        # test
        sss = [new_times[i].second + new_times[i].microsecond / 1000000.0 for i in range(4)]
        dt_test = np.array(sss[1:4]) - np.array(sss[0:3])
        if (dt_test - dt_last_lst >= 1e-3).all():
            print("ERROR in YAPS!!!!!!!!!!!")
        asdf = 1

    # new_transmitter_name[-1] = "A96-1111-12345"
    # new_transmitter_name[-2] = "A96-1111-12345"
    df_directions = pd.DataFrame({'Date and Time (UTC)': new_date_and_time_lst,
                                  'Receiver': new_receiver_name_lst,
                                  'Transmitter': new_transmitter_name,
                                  'Transmitter Name': None,
                                  'Transmitter Serial': None,
                                  'Sensor Value': None,
                                  'Sensor Unit': None,
                                  'Station Name': None,
                                  'Latitude': None,
                                  'Longitude': None})
    new_time = np.min(new_date_and_time_lst) - timedelta(microseconds=(np.min(new_date_and_time_lst)).microsecond)
    max_time = np.max(new_date_and_time_lst) - timedelta(microseconds=(np.max(new_date_and_time_lst)).microsecond)
    new_time -= timedelta(minutes=10)
    while new_time <= max_time:
        new_time += timedelta(minutes=10)
        for i in range(len(receiver_name_lst)):
            for val in [("A96-1111-12345", mean_point1),
                        ("A96-1111-12346", mean_point2),
                        ("A96-1111-12347", mean_point3)]:
                val_name, mean_point = val
                add_time = 1000000.0 * (np.linalg.norm(mean_point - np.array([x_lst[i], y_lst[i]])) / 1503.0)
                save_time = new_time + timedelta(microseconds=add_time)
                df_directions = pd.concat([df_directions,
                           pd.DataFrame.from_records({'Date and Time (UTC)': save_time,
                                                      'Receiver': receiver_name_lst[i],
                                                      'Transmitter': val_name}, index=[0])], ignore_index=True)

    df_hydros.to_csv(full_path + "yaps/" + "hydros.csv", index_label=False, index=False)
    df_directions.to_csv(full_path + "yaps/" + "directions.csv", index_label=False, index=False)

    mean_point = np.mean(np.array(rover_utm_lst), axis=0, dtype=int)
    str_lst = [plot_param_to_str(mean_point)] + str_lst

    with open(full_path + "test" + str(rover_file_num) + ".txt", 'w') as f:
        f.writelines(str_lst)

    # with open(full_path + "debug/" + "NextLocation" + str(rover_file_num) + ".pkl", 'wb') as f:
    #     pickle.dump(next_location_lst, f)

    with open(full_path + "debug/" + "TargetLocation" + str(rover_file_num) + ".pkl", 'wb') as f:
        pickle.dump(target_location_lst, f)


if __name__ == '__main__':
    # Example for old files:
    # convert_pkl(rover_file_num=75, target_file_num=75, is_static=True, new_depth=False, old_files=True)

    # Example for new files:
    convert_pkl(rover_file_num=1663738952,
                target_file_num=1663745933,
                is_static=False,
                new_depth=True,
                old_files=False,
                calculate_only_target_data=False)

    # 1663571619 1663562705
import ffmpeg
import numpy as np
from numpy.linalg import norm
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.pyplot as plt
from utils.time_utils import calculate_time_difference


# We present in a separate graph the calculation of the position of the object.
# We show the TDOA (hyperbolas) between any two rover locations t and t+1.
def plot_calc(ax, fig, rover_lst, t_lst, fish_pos, num_hyper, x_lim=None, y_lim=None, target=None, show_title=False):
    if show_title:
        ax.set_title('Calculation of the target location (with a visual representation DTOA of last ' + str(
            num_hyper + 1) + ' inputs)')
        ax.set_xlabel('x_axis (meters)')
        ax.set_ylabel('y_axis (meters)')

    if x_lim is None:
        x_lim = [-30, 30]
        y_lim = [-30, 30]
    points_lst = np.array(rover_lst)

    # We only show the last n locations of the rover.
    n_locations = points_lst.shape[0] - num_hyper - 1
    for i in np.arange(n_locations, points_lst.shape[0] - 1):
        label = None
        if i == n_locations:
            label = "Rover last locations"
        ax.plot(points_lst[i:i + 2, 0], points_lst[i:i + 2, 1], c='gray', linewidth=2, zorder=-1)
        ax.scatter(points_lst[i, 0], points_lst[i, 1], s=150, zorder=1, c='royalblue', label=label, marker=None)
        ax.scatter(points_lst[i + 1, 0], points_lst[i + 1, 1], s=150, zorder=1, c='royalblue', marker=None)
        ax.text(points_lst[i, 0] + .4, points_lst[i, 1] + .6, str(i - n_locations + 1), fontsize=15, zorder=2)
        ax.text(points_lst[i + 1, 0] + .4, points_lst[i + 1, 1] + .6, str(i - n_locations + 2), fontsize=15, zorder=2)

    if len(fish_pos) > 0:
        fish_pos = np.array(fish_pos)
        ax.scatter(fish_pos[0], fish_pos[1], marker='+', s=300, c='orange', linewidths=5,
                   label="Estimated target location", zorder=2)  # marker='+'
        print(fish_pos)

    if not (target is None):
        ax.scatter(target[0], target[1], marker='x', s=200, c='r', label="Real Target Location",
                   zorder=2, linewidths=5)

    # plot tdoa (hyperbolas) between any two rover locations t and t+1.
    if num_hyper == 2:
        color_lst = ['black', 'gray']  # ['limegreen', 'darkgreen']
    elif num_hyper == 3:
        color_lst = ['black', 'gray', 'lightgrey']  # ['greenyellow', 'limegreen', 'darkgreen']
    elif num_hyper == 4:
        color_lst = ['yellow', 'greenyellow', 'limegreen', 'darkgreen']
    elif num_hyper == 5:
        color_lst = ['gold', 'yellow', 'greenyellow', 'limegreen', 'darkgreen']
    elif num_hyper == 6:
        color_lst = ['orange', 'gold', 'yellow', 'greenyellow', 'limegreen', 'darkgreen']
    else:
        color_lst = ['pink'] * (num_hyper - 1) + ['royalblue']

    for i in np.arange(start=len(rover_lst) - num_hyper - 1, stop=len(rover_lst) - 1, step=1):
        color = color_lst[i - (len(rover_lst) - num_hyper - 1)]
        plot_hyperbola(ax, rover_lst[i:i + 2], t_lst[i:i + 2], 1500, color=color, n_count=i - n_locations + 1)

    ax.grid()
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    scale_bar = ScaleBar(1, location="lower right", width_fraction=0.02, pad=0.1, border_pad=0.1, length_fraction=0.25)
    ax.add_artist(scale_bar)

    # ax.set_box_aspect(1)
    fig.show()


# The function plot the hyperbola and calculate the parameters which are necessary for this task
def plot_hyperbola(ax, rov_lst, t_lst, velocity, color='pink', n_count=1):
    x_linspace, y_linspace = rov_lst[-1, 0:2]
    x, y = np.linspace(x_linspace - 200, x_linspace + 200, 2000), np.linspace(y_linspace - 200, y_linspace + 200, 2000)
    x, y = np.meshgrid(x, y)

    p_a, p_b = np.array(rov_lst)
    time_dif = calculate_time_difference(t_lst)  # time_mod10(t_lst)
    rd_ba = velocity * time_dif

    plot_hyperbola_contour(ax, x, y, p_a, p_b, rd_ba, color, n_count)


# The function plot the hyperbola given the necessary parameters for this task
def plot_hyperbola_contour(ax, x, y, p_a, p_b, rd_ba, colors, n_count=1):
    canonical_hyperbola_b_value = (norm(p_a - p_b) / 2) ** 2 - (abs(rd_ba) / 2) ** 2
    is_hyperbola = (canonical_hyperbola_b_value > 0)

    dis_b = ((x - p_b[0]) ** 2 + (y - p_b[1]) ** 2) ** 0.5
    dis_a = ((x - p_a[0]) ** 2 + (y - p_a[1]) ** 2) ** 0.5
    res = dis_b - dis_a - rd_ba

    label = "TDoA R" + str(n_count) + ",R" + str(n_count + 1) + ""

    # If the "b" value of canonical hyperbola is negative (which is not possible in reality),
    # we will print an approximation to the hyperbola
    if not is_hyperbola:
        res = np.round(np.abs(res), 1)
        label += " (approximate!)"

    hyper = ax.contour(x, y, res, [0], colors=colors, linestyles='dashed', linewidths=3, zorder=-1)
    # hyper.collections[0].set_label(label)
    ax.legend()


# This function is able to print all possible hyperbolas between two foci.
# TODO: check if we need to do similar approach to plot_hyperbola_contour function
def hyperbola_contour_all_options(ax, x, y, rd_ac, rd_ab, alpha, a_acx, b_acy, a_abx, b_aby, c_ab, c_ac):
    cos_a = np.cos(alpha)
    sin_a = np.sin(alpha)

    # plots all hyperbolas ___)_____(___
    if rd_ac != 0:
        ax.contour(x, y, (x ** 2 / a_acx ** 2 - y ** 2 / b_acy), [1], colors='k', linewidths=None)
    if rd_ab != 0:
        ax.contour(x, y, (((x + c_ac) * cos_a - y * sin_a - c_ab) ** 2 / a_abx ** 2 - (
                (x + c_ac) * sin_a + y * cos_a) ** 2 / b_aby), [1], colors='k', linewidths=None)


# Resets the map and plot the location history of the rover and the target.
def clean_path(ax, maps, rover_lst, fish_pos_lst, x0, y0, args, scale_bar):
    plot_xyArr, plot_fish_loc = None, None
    if len(rover_lst) > 0 and args.plot_prev_rover:
        plot_xyArr = np.array(rover_lst) + np.array([x0, y0, 0])
    if len(fish_pos_lst) > 0 and args.plot_prev_target:
        plot_fish_loc = np.array(fish_pos_lst) + np.array([x0, y0, 0])
    maps.reset_plot(ax, scale_bar, plot_xyArr, plot_fish_loc)


# Converts a series of images into a video (the video supports any platform)
def mp4_ffmpeg(srt_input, str_output, framerate=0.5, isHD=False):
    stream = ffmpeg.input(srt_input, pattern_type='glob', framerate=framerate)
    # stream = stream.filter('scale', size='hd1080', force_original_aspect_ratio='increase')
    if isHD:
        stream = stream.filter('scale', size='hd1080')
    else:
        stream = stream.filter('scale', 1080, -1)
    stream = stream.output(str_output, pix_fmt='yuv420p')  # , **{'b:v': '64k'})
    stream.run()


def distance_error_no_plot(fish_lst, target_real_loc_lst, travel_report_lst, x0=0, y0=0, i=0, n_cut=0):
    fish_lst = np.array(fish_lst)  # + np.array([x0, y0, 0])
    n = fish_lst.shape[0]
    target_real_loc_lst = np.array(target_real_loc_lst[i:n + i])

    xy_error = np.linalg.norm(fish_lst[:, 0:2] - target_real_loc_lst[:, 0:2], axis=1)
    xy_error = (xy_error[travel_report_lst])[0:n - n_cut]
    xy_error_mean = np.around(np.mean(xy_error), decimals=2)

    xy_error_rms = np.around(np.power(np.sum(np.power(xy_error, 2)) / len(xy_error), 0.5), decimals=2)
    xy_error_median = np.around(np.median(xy_error), decimals=2)
    xy_error_std = np.around(np.std(xy_error), decimals=2)

    return xy_error, xy_error_mean, xy_error_rms, xy_error_median, xy_error_std


# In case we have the true locations of the target, we will compare the distance error.
def distance_error_plot(fish_lst, target_real_loc_lst, travel_report_lst, x0=0, y0=0, i=0, n_cut=0):
    xy_error, xy_error_mean, xy_error_rms, xy_error_median, xy_error_std = \
        distance_error_no_plot(fish_lst, target_real_loc_lst, travel_report_lst, x0, y0, i, n_cut)

    fig, ax = plt.subplots(1, 2, figsize=(16.0, 32.0))

    ax[0].plot(xy_error)
    ax[0].set_title('Distance error (' + 'mean ' + str(xy_error_mean) + ', ' +
                    'RMS ' + str(xy_error_rms) + ', ' +
                    'median ' + str(xy_error_median) + ', ' +
                    'std ' + str(xy_error_std) + ')')
    ax[0].set_xlabel('Experiment number')
    ax[0].set_ylabel('Error size (meter)')
    ax[0].set_ylim([0, 20])

    ax[1].hist(xy_error, bins=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 100])
    ax[1].set_title('Localization error Histogram')
    ax[1].set_xlabel('Error size (meter)')
    ax[1].set_ylabel('')
    ax[1].set_xlim([0, 22])
    ax[1].set_ylim([0, 20])
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    fig.tight_layout()
    ax[0].set_box_aspect(1)
    ax[1].set_box_aspect(1)


def error_plot_for_list(data_lst, filter_lst, title, unit_of_measurement, bins,
                        ylim, n_cut=0, is_polar=False, angle_bins=3):
    data_lst = np.array(data_lst)
    n = data_lst.shape[0]

    data_error = data_lst[filter_lst][0:n - n_cut]
    data_mean = np.around(np.mean(data_error), decimals=2)
    data_median = np.around(np.median(data_error), decimals=2)
    data_std = np.around(np.std(data_error), decimals=2)

    if is_polar:
        fig, ax = plt.subplots(1, 2, subplot_kw=dict(projection='polar'), figsize=(16.0, 10.0))
        ax[1].set_theta_zero_location("N")
        ax[0] = fig.add_subplot(1, 2, 1)
    else:
        fig, ax = plt.subplots(1, 2, figsize=(16.0, 10.0))

    ax[0].plot(data_error)
    ax[0].set_title(title + ' (' + 'mean ' + str(data_mean) + ', ' +
                    'median ' + str(data_median) + ', ' +
                    'std ' + str(data_std) + ')')
    ax[0].set_xlabel('Experiment number')
    ax[0].set_ylabel('Error size in ' + unit_of_measurement)
    ax[0].set_ylim(ylim)

    if is_polar:
        # Visualise by area of bins
        circular_hist(ax[1], data_error, bins=angle_bins, density=False, offset=0)
        fig.tight_layout()
    else:
        ax[1].hist(data_error, bins=bins)
        ax[1].set_title('Histogram of ' + title)
        ax[1].set_xlabel('Error size in ' + unit_of_measurement)
        ax[1].set_ylabel('count')
        ax[1].set_xlim([bins[0], bins[-1] + 5])
        ax[1].set_ylim([0, 50])
        fig.subplots_adjust(hspace=0.5, wspace=0.5)
        fig.tight_layout()


# https://stackoverflow.com/questions/22562364/circular-polar-histogram-in-python
def circular_hist(ax, data, bins=16, density=True, offset=0, gaps=True):
    """
    Produce a circular histogram of angles on ax.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.PolarAxesSubplot
        axis instance created with subplot_kw=dict(projection='polar').

    x : array
        Angles to plot, expected in units of radians.

    bins : int, optional
        Defines the number of equal-width bins in the range. The default is 16.

    density : bool, optional
        If True plot frequency proportional to area. If False plot frequency
        proportional to radius. The default is True.

    offset : float, optional
        Sets the offset for the location of the 0 direction in units of
        radians. The default is 0.

    gaps : bool, optional
        Whether to allow gaps between bins. When gaps = False the bins are
        forced to partition the entire [-pi, pi] range. The default is True.

    Returns
    -------
    n : array or list of arrays
        The number of values in each bin.

    bins : array
        The edges of the bins.

    patches : `.BarContainer` or list of a single `.Polygon`
        Container of individual artists used to create the histogram
        or list of such containers if there are multiple input datasets.
    """
    ax.set_theta_zero_location("N")
    x = np.copy(data)
    # x = np.array(x)
    # x[x < 0] = 360 + x[x < 0]

    # Wrap angles to [-pi, pi)
    x = (x / 180.0) * np.pi
    # x = (x + np.pi) % (2 * np.pi) - np.pi

    # Force bins to partition entire circle
    if not gaps:
        bins = np.linspace(-np.pi, np.pi, num=bins + 1)

    # Bin data and record counts
    n, bins = np.histogram(x, bins=bins)

    # Compute width of each bin
    widths = np.diff(bins)

    # By default plot frequency proportional to area
    if density:
        # Area to assign each bin
        area = n / x.size
        # Calculate corresponding bin radius
        radius = (area / np.pi) ** .5
    # Otherwise plot frequency proportional to radius
    else:
        radius = n

    # Plot data on ax
    patches = ax.bar(bins[:-1], radius, zorder=1, align='edge', width=widths,
                     edgecolor='C0', fill=True, linewidth=3)

    # Set the direction of the zero angle
    ax.set_theta_offset(offset)

    # Remove ylabels for area plots (they are mostly obstructive)
    if density:
        ax.set_yticks([])

    return n, bins, patches

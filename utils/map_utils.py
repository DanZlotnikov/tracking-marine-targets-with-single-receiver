import os
import shutil
import pickle
import numpy as np
import matplotlib.image as plt_img
import matplotlib.pyplot as plt

from utm import from_latlon, to_latlon
from maps.bing_maps.download_map import download_tiles, stitching_tiles


# The purpose of the function is to load a map.
# If the map does not exist locally, we will download
# a new map according to the map's top right/left and bottom left/right coordinates.
def create_map(lt_lat: float, lt_lng: float, rb_lat: float, rb_lng: float, map_name: str, style: str):
    path = "maps/" + map_name
    if not os.path.isdir(path):
        maps = MapClass(lt_lat, lt_lng, rb_lat, rb_lng, map_name, style=style)
        with open(path + "/map_param.pkl", 'wb') as map_output:
            pickle.dump(maps, map_output, -1)
    else:
        with open(path + "/map_param.pkl", 'rb') as map_input:
            maps = pickle.load(map_input)
    maps.set_img_map()
    return maps


# Set display boundaries of the map
def set_lim(ax, p_x, p_y, lim=100):
    ax.set_xlim([p_x - lim, p_x + lim])
    ax.set_ylim([p_y + lim, p_y - lim])


class MapClass:
    def __init__(self, lt_lat: float, lt_lng: float, rb_lat: float, rb_lng: float, map_name: str, style: str):
        # Download the map tiles, merge them, and save them in the maps folder.
        path = "maps/" + map_name
        os.mkdir(path)
        download_tiles(lt_lat, lt_lng, rb_lat, rb_lng, style=style)
        stitching_tiles(map_name)
        shutil.rmtree("maps/Images")

        # we save map's top and bottom coordinates
        self.zone_num = int(from_latlon(lt_lat, lt_lng)[2])
        self.zone_letter = from_latlon(lt_lat, lt_lng)[3]
        top_point_utm = np.array(from_latlon(lt_lat, lt_lng)[0:2], dtype=float)
        bottom_point_utm = np.array(from_latlon(rb_lat, rb_lng)[0:2], dtype=float)

        # We save the image details of the map, but we do not keep the image itself within the class.
        self.map_name = map_name
        img_map = plt_img.imread(path + '/ArielView.jpeg')
        self.img_shape = np.array([img_map.shape[1], img_map.shape[0], img_map.shape[2]])
        #  Only after the creation of the class and save it locally,
        #  we uploading the image into the class.
        self.img_map = None

        self.x_length = np.abs(top_point_utm[0] - bottom_point_utm[0])
        self.y_length = np.abs(top_point_utm[1] - bottom_point_utm[1])

        self.pixel_length = self.x_length / self.img_shape[0]

        # we save the top right coordinate of the map
        self.top_right_utm = np.max(np.array([bottom_point_utm, top_point_utm]), 0)
        self.bottom_left_utm = np.min(np.array([bottom_point_utm, top_point_utm]), 0)
        self.mean_point = None

    # After each calculation of the target location,
    # we reset the graphs and display the latest Rover and target locations.
    def reset_plot(self, ax, scale_bar, rover_pos=None, fish_pos=None, show_title=False):
        ax.clear()
        if show_title:
            ax.set_title('Map of the ' + self.map_name + ' area (UTM)')
            ax.set_xlabel('x_axis (meters)')
            ax.set_ylabel('y_axis (meters)')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.imshow(self.img_map, aspect="auto")
        ax.add_artist(scale_bar)

        if rover_pos is not None:
            point_plot = self.plot_point(rover_pos, ax, None, 150, 'purple', True)
            point_plot.set_label("Previous Rover locations where received a signal")
            ax.legend()

        if fish_pos is not None:
            point_plot = self.plot_point(fish_pos, ax, None, 150, 'pink', False)
            point_plot.set_label("Previous target locations")
            ax.legend()

    # Given the coordinate, we display it on the map.
    def plot_point(self, p_plot, ax, marker, s, c=None, set_xy_lim=True,
                   width=None, plot_lim=50, is_line=False, line_label=""):
        p_plot_x, p_plot_y = self.convert_point(p_plot)
        if is_line:
            point_plot = ax.plot(p_plot_x, p_plot_y, c=c, linewidth=width, label=line_label)
        else:
            point_plot = ax.scatter(p_plot_x, p_plot_y, marker=marker, s=s, c=c, linewidths=width)

        if self.mean_point is None:
            x_point, y_point = p_plot_x[-1], p_plot_y[-1]
        else:
            x_point, y_point = self.convert_point(self.mean_point[0:2])

        if set_xy_lim:
            set_lim(ax, x_point, y_point, lim=plot_lim)
        return point_plot

    # A coordinate converter to a new coordinate on the map.
    def convert_point(self, p_plot):
        p_plot_x = (1-(self.top_right_utm[0] - p_plot[:, 0]) / self.x_length) * self.img_shape[0]
        p_plot_y = ((self.top_right_utm[1] - p_plot[:, 1]) / self.y_length) * self.img_shape[1]

        return p_plot_x, p_plot_y

    # Converts from lat&lon cataracts to UTM coordinates
    @staticmethod
    def map_proj_utm(point_lat, point_lon):
        return np.array(from_latlon(point_lat, point_lon)[0:2], dtype=float)

    # Converts from UTM coordinates to lat&lon cataracts
    def map_proj_ll(self, point_utm_x, point_utm_y):
        return np.array(to_latlon(point_utm_x, point_utm_y, self.zone_num, self.zone_letter)[0:2], dtype=float)

    # Uploads the image to the class
    def set_img_map(self):
        path = "maps/" + self.map_name + '/ArielView.jpeg'
        self.img_map = plt_img.imread(path)

    # plot the real location of (only static at this time) target
    def plot_real_target(self, ax, target_loc, is_line=False):
        point_plot = self.plot_point(target_loc, ax, None, s=150, c='r',
                                     set_xy_lim=False, width=5, is_line=is_line, line_label="Real Target Locations")
        if not is_line:
            point_plot.set_label("Real target location")
        ax.legend()

    # plot the real location of (only static at this time) target
    def plot_rover_path(self, ax, rover_path_lst):
        point_plot = self.plot_point(rover_path_lst, ax, '*', s=50, c='w', set_xy_lim=False, width=5)
        point_plot.set_label("Rover locations")
        ax.legend()

    # plot the real location of (only static at this time) target
    def plot_estimate_target(self, ax, target_loc_lst, is_line=False):
        point_plot = self.plot_point(target_loc_lst, ax, None, s=150, c='orange',
                                     set_xy_lim=False, width=5, is_line=is_line,
                                     line_label="Calculated target Locations")
        if not is_line:
            point_plot.set_label("Calculated target Locations")
        ax.legend()

    # plot the real location of (only static at this time) target
    def plot_target_echo(self, ax, target_echo_lst):
        point_plot = self.plot_point(target_echo_lst, ax, None, s=150, c='royalblue',  set_xy_lim=False, width=5)
        point_plot.set_label("Received signals from tag")
        ax.legend()


    def end_of_the_task(self, mean_point, ax, is_plot_real_target_location, target_real_lst,
                        target_estimate_lst, check_poly_lst, lim=400, with_dots=False):

        p_plot_x, p_plot_y = self.convert_point(np.array([mean_point]))
        set_lim(ax, p_plot_x[-1], p_plot_y[-1], lim=lim)

        target_estimate_lst = target_estimate_lst[check_poly_lst]
        target_real_lst = target_real_lst[check_poly_lst]

        if is_plot_real_target_location:
            self.plot_real_target(ax, target_real_lst, is_line=True)
        self.plot_estimate_target(ax, target_estimate_lst, is_line=True)

        if with_dots:
            self.end_of_the_task_with_dots(ax, is_plot_real_target_location, target_real_lst, target_estimate_lst)

    def end_of_the_task_with_dots(self, ax, is_plot_real_target_location, target_real_lst, target_estimate_lst):
        for i in range(target_estimate_lst.shape[0]):
            color = np.random.rand(3, )
            target_estimate_lst_i = self.convert_point(np.array([target_estimate_lst[i][:2]]))
            if is_plot_real_target_location:
                target_real_lst_i = self.convert_point(np.array([target_real_lst[i][:2]]))
                p_plot_x = np.array([target_estimate_lst_i[0], target_real_lst_i[0]])[:, 0]
                p_plot_y = np.array([target_estimate_lst_i[1], target_real_lst_i[1]])[:, 0]
                # ax.plot(p_plot_x, p_plot_y, c='w', linewidth=3)
                ax.scatter(target_real_lst_i[0], target_real_lst_i[1], marker=None, s=150, c='r', linewidths=None)
            ax.scatter(target_estimate_lst_i[0], target_estimate_lst_i[1], marker=None, s=150, c='orange', linewidths=None)

    # def end_of_the_task2(self, p_plot, ax, is_plot_real_target_location,
    #                      target_real_lst, rover_path_lst, target_estimate_lst,
    #                      target_echo_lst, check_poly_lst, radius_lst):
    #     p_plot_x, p_plot_y = self.convert_point(np.array([p_plot]))
    #     set_lim(ax, p_plot_x[-1], p_plot_y[-1], lim=400)
    #     # ax.set_box_aspect(1)
    #
    #     target_estimate_lst = target_estimate_lst[check_poly_lst]
    #     target_real_lst = target_real_lst[:-3][check_poly_lst]
    #     radius_lst = radius_lst[:-3][check_poly_lst]
    #
    #     for i in range(target_estimate_lst.shape[0]):
    #         color = np.random.rand(3, )
    #         target_estimate_lst_i = self.convert_point(np.array([target_estimate_lst[i][:2]]))
    #         target_real_lst_i = self.convert_point(np.array([target_real_lst[i][:2]]))
    #
    #         p_plot_x = np.array([target_estimate_lst_i[0], target_real_lst_i[0]])[:, 0]
    #         p_plot_y = np.array([target_estimate_lst_i[1], target_real_lst_i[1]])[:, 0]
    #         ax.plot(p_plot_x, p_plot_y, c=color, linewidth=3)
    #
    #         ax.scatter(target_estimate_lst_i[0], target_estimate_lst_i[1], marker='+', s=120, c='orange', linewidths=3)
    #
    #         target_estimate_lst_i = self.convert_point(np.array([target_estimate_lst[i][:2]]))
    #         circle = plt.Circle((target_estimate_lst_i[0], target_estimate_lst_i[1]), radius_lst[i], fill=False, color=color)
    #         ax.add_patch(circle)


    def end_of_the_task3(self, p_plot, ax, is_plot_real_target_location,
                         target_real_lst, rover_path_lst, target_estimate_lst,
                         target_echo_lst, check_poly_lst, radius_lst):
        p_plot_x, p_plot_y = self.convert_point(np.array([p_plot]))
        set_lim(ax, p_plot_x[-1], p_plot_y[-1], lim=400)
        # ax.set_box_aspect(1)

        n = np.min([target_estimate_lst.shape[0],target_real_lst.shape[0],radius_lst.shape[0]])
        target_estimate_lst = target_estimate_lst[:n]# [check_poly_lst]
        target_real_lst = target_real_lst[:n] # target_real_lst[:-3][check_poly_lst]
        radius_lst = radius_lst[:n] #radius_lst[:-3][check_poly_lst]

        for i in range(target_estimate_lst.shape[0]):
            color = np.random.rand(3, )
            target_estimate_lst_i = self.convert_point(np.array([target_estimate_lst[i][:2]]))
            target_real_lst_i = self.convert_point(np.array([target_real_lst[i][:2]]))

            p_plot_x = np.array([target_estimate_lst_i[0], target_real_lst_i[0]])[:, 0]
            p_plot_y = np.array([target_estimate_lst_i[1], target_real_lst_i[1]])[:, 0]
            ax.plot(p_plot_x, p_plot_y, c=color, linewidth=None)

            ax.scatter(target_estimate_lst_i[0], target_estimate_lst_i[1], marker=None, s=150, c='orange', linewidths=None)

            target_estimate_lst_i = self.convert_point(np.array([target_estimate_lst[i][:2]]))
            circle = plt.Circle((target_estimate_lst_i[0], target_estimate_lst_i[1]), radius_lst[i], fill=False, color=color)
            ax.add_patch(circle)
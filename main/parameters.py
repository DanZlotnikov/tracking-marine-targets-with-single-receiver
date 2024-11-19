import os
import argparse
import matplotlib
import numpy as np

# font = {'family': 'normal',
#         'weight': 'bold',
#         'size': 7}
#
# matplotlib.use("TkAgg")
# matplotlib.rc('font', **font)

import matplotlib.pyplot as plt

current_path = os.path.dirname(os.path.abspath(__file__))

SMALL_SIZE = 15
MEDIUM_SIZE = 15
BIGGER_SIZE = 15

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# In this file we define all the parameters common to the whole program
parser = argparse.ArgumentParser(description='rover_params')

parser.add_argument('--current-path', type=str, default=current_path, metavar='N', 
                    help='current path of the program')

####################
# rover parameters #
####################
parser.add_argument('--velocity', type=int, default=1500, metavar='N',
                    help='Velocity of the rover m/sec')

##################
# map parameters #
##################
# Eilat_IUI - 29.506496, 34.920970, 29.497808, 34.914200
# Beer_Sheva_Lake - 31.235509, 34.821957, 31.238811, 34.828061
parser.add_argument('--map-name', type=str, default="Eilat_IUI", metavar='N',
                    help='Select an existing or new map')
parser.add_argument('--map-cod', type=list, default=[29.506496, 34.920970, 29.497808, 34.914200], metavar='N',
                    help='left top and right bottom coordinates of the map')
# parser.add_argument('--map-name', type=str, default="Ashdod", metavar='N',
#                     help='Select an existing or new map')
# parser.add_argument('--map-cod', type=list, default=[31.804094, 34.643694, 31.790997, 34.653294], metavar='N',
#                     help='left top and right bottom coordinates of the map')
parser.add_argument('--map-style', type=str, default='r', metavar='N',
                    help='r = road, a = aerial, h = hybrid')


######################
# program parameters #
######################
parser.add_argument('--num-file', type=int, default=1663745933, metavar='N',
                    help='Select a sample file of acoustic waves')  # 75 + Static, 1659624344 + Dynamic
# parser.add_argument('--num-file-target', type=int, default=1654159170+1, metavar='N',
#                     help='Select a target real location file')
parser.add_argument('--min-distance', type=int, default=10, metavar='N',
                    help='Minimum distance between any two Rover locations that have received the tag input (meters)')
parser.add_argument('--target-real-location', type=str, default='Dynamic', metavar='N',
                    help='Use the original position of the target: None/Static/Dynamic')
parser.add_argument('--convert-coordinate-utm', type=str, default='NORMAL', metavar='N',
                    help='file type: GPGGA or NORMAL')

##########################
# calculation parameters #
##########################
parser.add_argument('--tdoa-type', type=str, default="tdoa_pairs", metavar='N',
                    help='Algorithm type for calculating target location: classic_tdoa, optimization_tdoa, '
                         'heat_map_tdoa, heat_map_distance, tdoa_pairs')
parser.add_argument('--is-3d', type=bool, default=False, metavar='N',
                    help='Algorithm can use the target depth')
parser.add_argument('--shape', type=str, default="4", metavar='N',
                    help='Algorithm type for calculating shape moving location: triangle_only, int >= 3')
parser.add_argument('--max-distance', type=int, default=20, metavar='N',
                    help='The maximum distance at which the rover can move from its last position to the next')
parser.add_argument('--radius-distance', type=int, default=10, metavar='N',
                    help='The distance at which the rover should be from the target')
parser.add_argument('--sim-path', type=str, default=current_path, metavar='N', 
                    help='where to dave files of the simulation')

###################
# plot parameters #
###################
parser.add_argument('--show-rpi-calc', type=bool, default=False, metavar='N',
                    help='show what only rpi calculate')
parser.add_argument('--show-calc-plot', type=bool, default=True, metavar='N',
                    help='show side by side map plot and calc plot')
parser.add_argument('--plot-prev-rover', type=bool, default=False, metavar='N',
                    help='Print previous Rover locations')
parser.add_argument('--plot-prev-target', type=bool, default=False, metavar='N',
                    help='Print previous target locations calculated by the program')
parser.add_argument('--lim-plot', type=int, default=150, metavar='N',
                    help='set x,y legends length to be 2xn')
parser.add_argument('--set-legends-map', type=bool, default=True, metavar='N',
                    help='show legends on the maps plots')
parser.add_argument('--set-legends-calc', type=bool, default=True, metavar='N',
                    help='show legends on the tdoa calculations plots')

#########################
# simulators parameters #
#########################
parser.add_argument('--time-accuracy', type=int, default=3, metavar='N',
                    help='How many digits accuracy in the times we require')
parser.add_argument('--maxIter', type=int, default=360, metavar='N',
                    help='How many steps we would like to do')

parser.add_argument('--target-start-location', type=list, default=[0, 0, -1], metavar='N',
                    help='From where the target starts the path')
parser.add_argument('--rover-start-location', type=list, default=[[-5, -2, 0], [6, -3, 0]], metavar='N',
                    help='From where the rover starts the search')

parser.add_argument('--plot-simulators', type=bool, default=False, metavar='N',
                    help='From where the rover starts the search')

parser.add_argument('--use-dronekit-sitl', type=bool, default=False, metavar='N',
                    help='Allows simulation of sending commands to vehicles in addition to simulation')

parser.add_argument('--use-interception', type=bool, default=True, metavar='N',
                    help='Allows simulation intercept the signals with 50 percentage of success')

parser.add_argument('--always-get-update-on-calculation-error', type=bool, default=False, metavar='N',
                    help='Do we want that at every step of the run we save our summary of calculation errors'
                         '(at the end of the program it is printed anyway)')

parser.add_argument('--t-delta', type=float, default=(1.0 / 1000.0), metavar='N',
                    help='The size of the error added to the arrival time of acoustic input')  # 0.5/1.0/1.5
parser.add_argument('--loc-delta', type=float, default=np.sqrt(0), metavar='N',
                    help='The size of the error added to current position of the rover')  # 1-3
parser.add_argument('--next-loc-delta', type=float, default=np.sqrt(0), metavar='N',
                    help='The size of the error added to the next rover position')  # 1-3

parser.add_argument('--sim-name', type=str, default='4_toa_loc_delta_0', metavar='N',
                    help='The size of the error added to the next rover position')

args = parser.parse_args()

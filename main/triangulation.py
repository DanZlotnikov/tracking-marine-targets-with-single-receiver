import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

from target_position_functions.calculation_tdoa_optimization import *
from utm import *



# Main Function
if __name__ == "__main__":
    # Target Location (Lat, Lon)
    target_lat, target_lon = 31.261920466113672, 34.80374511656835

    # Listener Locations (Lat, Lon)
    listener_positions = [
        (31.26208075293989, 34.80430923818613),
        (31.262293710131743, 34.80378330966735),
        (31.262045500327204, 34.80339194107266)
    ]

    # Speed of signal propagation in air (m/s)
    velocity = 343

    # Convert Target Position and Add Depth (z = 0)
    target_utm_x, target_utm_y, zone_number, zone_letter = from_latlon(target_lat, target_lon)
    target_utm = [target_utm_x, target_utm_y, 0]  # Add z = 0

    # Convert Listener Positions and Add Depth (z = 0)
    listener_utm = [
        list(from_latlon(lat, lon)[:2]) + [0]  # Add z = 0
        for lat, lon in listener_positions
    ]

    # Calculate Distances and Time of Arrival (TOA)
    distances = [np.linalg.norm(np.array(listener) - np.array(target_utm)) for listener in listener_utm]
    toa = [distance / velocity for distance in distances]

    # Initial Guess for Target Position in 3D
    initial_guess = np.array(target_utm) + np.array([10, -10, 0])  # Offset only in x, y; z remains 0

    # Call Steepest Descent
    rov_lst = np.array(listener_utm)  # Listeners' positions
    t_lst = np.array(toa)  # TOA values
    result, failed = steepest_descent(rov_lst, t_lst, velocity, initial_guess, is_2d=True)

    if not failed:
        # Convert Result Back to Lat/Lon
        estimated_latlon = to_latlon(result[0][0], result[0][1], zone_number, zone_letter)
        print("Estimated Target Position (Lat/Lon):", estimated_latlon)
    else:
        print("Steepest Descent Failed")
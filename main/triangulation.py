import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

from target_position_functions.calculation_tdoa_classic import *

if __name__ == '__main__':
    print(calc_parm(1,2,3,4))
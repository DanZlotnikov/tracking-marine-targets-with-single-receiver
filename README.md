# Acoustic tracking of moving marine targets using a single autonomous surface receiver

## Introduction

While marine animal behavior is often studied in constrained lab setups, a more reliable exploration should be done in their natural environment and without human interference. This task becomes excessively more challenging when quantitative data are needed in large and unconstrained aquatic environments. Toward that end, researchers widely use acoustic positioning telemetry to remotely track their subjects, though this often requires an extensive network of receivers placed in the environment ahead of time. This study proposes a new tracking method that continuously tracks and reports the trajectory of a target in unconstrained marine environments using a single-moving acoustic receiver. Instead of deploying an extensive array of static receivers, we use a single receiver mounted on an autonomous surface vehicle to obtain highly accurate results with much cheaper and simpler means. The receiver position and earlier target location estimations are used to calculate an optimal trajectory for the receiver, which in turn provides subsequent readings and target localizations based on a new variant of the Time Difference of Arrival approach. We demonstrate the performance of the proposed methods using both simulations and field experiments.

This repository contains the implementation of the methods proposed in this [study](https://onlinelibrary.wiley.com/doi/full/10.1002/rob.22335).

## Features

- **Rover Calculation for the Next Best Position**: Algorithms to determine the optimal next position for the rover based on current and previous data.
- **Target Localization**: Methods to accurately locate the target using Time Difference of Arrival (TDOA) and other advanced techniques.
- **Simulations**: Tools to simulate various scenarios and validate the performance of the tracking methods in different environments.

## Installation

### Prerequisites

- Python 3.9
- Anaconda or Miniconda (for environment management)
- Git (to clone the repository)

### Setup Instructions

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/fish-navigation.git
   cd fish-navigation
   ```

2. **Create a Conda Environment**

     ```bash
     conda env create -f Fish_Navigation.yaml
     ```

3. **Activate the Environment**

   ```bash
   conda activate Fish_Navigation
   ```

4. **Install Required Packages**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Simulator

Execute the simulation script to run simulations with your specified parameters:

```bash
python main/simulator.py
```

You can adjust simulation settings in `parameters.py` to customize the simulation runs.

### Post-Processing Simulation Data

For validate the filed test, use the post-processing script to analyze and visualize the results:
(Note: The repository does not include the data files; they will be added later)

```bash
python main/post_processing.py
```

## Folder Structure


### maps/

Stores map data downloaded using the map downloader.

### utils/

Utility functions and helper scripts used across the project.

#### Files:

- **main_utils.py**: Core utility functions used throughout the project.
- **map_utils.py**: Functions for handling map data and operations.
- **pkl_utils.py**: Utilities for handling pickle files for data serialization.
- **plots_utils.py**: Helper functions for generating and customizing plots.
- **quartic_solver_utils.py**: Contains solvers for quartic equations used in calculations.
- **simulator_utils.py**: Utilities that assist in running and managing simulations.
- **time_utils.py**: Functions for time calculations, including time differences.

### rover_position_functions/

Functions related to calculating the rover's position.

#### Files:

- **next_location_calculation.py**: Calculates the next location of the rover based on current parameters.
- **next_location_triangle_calculation.py**: Determines the rover's next position using triangle calculations.

### target_position_functions/

Functions for determining the target's (e.g., fish's) position.

#### Files:

- **calculation_distance_between_hyperbolas_heatmap.py**: Calculates distances between hyperbolas using heat maps.
- **calculation_tdoa_classic.py**: Implements classic TDOA calculations for target positioning.
- **calculation_tdoa_heatmap.py**: Uses heat maps for TDOA calculations to improve accuracy.
- **calculation_tdoa_optimization.py**: Provides optimized algorithms for TDOA calculations.

### Other Directories and Files

- **parameters.py**: Contains configurable parameters and settings for simulations.
- **main/show.py**: Script for visualizing results from previous experiments.
- **main/simulator.py**: Runs simulations and allows executing multiple simulations with combined results.
- **main/post_processing.py**: Processes and analyzes filed test data post-execution.
- **Fish_Navigation.yaml**: Environment setup file for Python 3 using Anaconda.

## License

This project is licensed under the terms of the [APGL 3.0](LICENSE.md).

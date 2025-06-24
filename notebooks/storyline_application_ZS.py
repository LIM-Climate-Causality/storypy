# This code demonstrates a complete analysis workflow using the `storypy` library.
# It processes a target climate variable and evaluates remote drivers, performs regression,
# and produces storyline-based visualizations.

import storypy  # Assumes storypy is installed and available
import xarray as xr
import pandas as pd

# -----------------------
# Step 1: Define Configuration Dictionaries
# -----------------------

# Configuration for the target variable
config_target = {
    "variable": "pr",  # example: near-surface zonal wind
    "region": {
        "lat_bounds": [-60, -30],
        "lon_bounds": [0, 360]
    },
    "reference_period": [1960, 1990],
    "anomaly_period": [2070, 2099],
    "season": "DJF",
    "work_folder": "./work"
}

# Configuration for the remote drivers
config_drivers = {
    "drivers": ["global_warming", "tropical_warming", "polar_amplification"],
    "boxes": {
        "lat_bounds": [-90, 90],
        "lon_bounds": [0, 360],
        "lat_bounds": [-15, 15],
        "lon_bounds": [0, 360],
        "lat_bounds": [-50, -60],
        "lon_bounds": [0, 360]
    },
    "reference_period": [1960, 1990],
    "anomaly_period": [2070, 2099]
}

# -----------------------
# Step 2: Generate Target Variable
# -----------------------

# This class processes the target variable and saves a NetCDF file and some figures
processor_target = storypy.TargetProcessor(config=config_target)
processor_target.generate_target_variable()

# -----------------------
# Step 3: Generate Remote Drivers
# -----------------------

# This class calculates the remote drivers and saves them to NetCDF
processor_drivers = storypy.RemoteDriverProcessor(config=config_drivers)
processor_drivers.generate_drivers()

# -----------------------
# Step 4: Evaluate Remote Drivers
# -----------------------

# This function evaluates correlation/statistical relevance of remote drivers
# and stores results in a CSV file
storypy.evaluate_remote_drivers(
    target_config=config_target,
    driver_config=config_drivers,
    output_csv_path="./work/remote_driver_evaluation.csv"
)

# -----------------------
# Step 5: Load Processed Data
# -----------------------

# Load target variable and remote drivers (assumes standard naming convention from storypy)
target = xr.open_dataarray("./work/target_variable.nc")
drivers = xr.open_dataset("./work/remote_drivers.nc")
evaluation_df = pd.read_csv("./work/remote_driver_evaluation.csv")

# -----------------------
# Step 6: Perform Multiple Linear Regression
# -----------------------

regression_result = storypy.multiple_linear_regression(target, drivers)
regression_result.to_netcdf("./work/regression_coefficients.nc")

# -----------------------
# Step 7: Storyline Evaluation
# -----------------------

# This function evaluates different storylines based on warming scenarios and regression coefficients
storylines = storypy.storyline_evaluation(
    regression_coefficients=regression_result,
    driver_data=drivers,
    level_of_warming=2.0,  # 2 degrees warming
    storyline_coeff=1.5    # amplifying coefficient for storyline impact
)

# -----------------------
# Step 8: Produce Figures
# -----------------------

# Generate regression maps
storypy.plot_regression_coefficients(regression_result, output_folder="./figures")

# Generate storyline maps
storypy.plot_storylines(storylines, output_folder="./figures")

# Generate warming levels
storypy.plot_level_of_warming(drivers, output_folder="./figures")

from ._regres import spatial_MLR
from storypy.utils import xr, pd
import os


# def run_regression(config: dict) -> list[str]:
#     """
#     Run spatial multiple linear regression (MLR) for multiple target variables using NetCDF and CSV data.

#     Parameters:
#         config (dict): Configuration with keys like 'work_dir'.
#         target_vars (list of str): List of variable names in the NetCDF dataset to use as regression targets.
#     """

#     target_vars = config.get("target_variable", [])
#     if not target_vars:
#         raise ValueError("No target variables specified in config['target_variable'].")

#     target_path = os.path.join(config['work_dir'], "target.nc")
#     driver_path = os.path.join(config['work_dir'], "driver_outputs/remote_drivers/scaled_standardized_drivers.csv")

#     ds = xr.open_dataset(target_path)
#     regressors = pd.read_csv(driver_path, index_col=0)
#     regressors.index = regressors.index.str.strip()

#     ds_unique = ds.groupby('model').first()
#     common_models = list(regressors.index.intersection(ds_unique['model'].values))

#     if not common_models:
#         raise ValueError("No common models found between NetCDF and CSV data.")

#     ds_subset = ds_unique.sel(model=common_models).reindex(model=common_models)
#     regressors_aligned = regressors.loc[common_models]
#     regressor_names = regressors_aligned.columns.insert(0, 'MEM')

#     MLR = spatial_MLR()
#     MLR.regression_data(None, regressors_aligned, regressor_names)

#     output_path = os.path.join(config["work_dir"], 'regression_output')
#     os.makedirs(output_path, exist_ok=True)

#     output_files = []

#     for var in target_vars:
#         if var not in ds_subset:
#             print(f"Warning: Variable '{var}' not found in dataset.")
#             continue

#         target = ds_subset[var]
#         MLR = spatial_MLR()
#         MLR.regression_data(target, regressors_aligned, regressor_names)

#         output_subdir = os.path.join(output_path, var)
#         os.makedirs(output_subdir, exist_ok=True)
    
#         output_file = MLR.perform_regression(output_path, var)
#         output_files.append(output_file)
#         print(f"Regression completed for: {var}")

#     return output_files


def run_regression(config: dict) -> list[str]:
    """
    Run spatial multiple linear regression (MLR) for a single target variable using NetCDF and CSV data.

    Parameters:
        config (dict): Must contain 'work_dir' and a single-element list in 'target_variable'.

    Returns:
        list of str: Paths to the regression output files.
    """

    # Validate config
    target_vars = config.get("target_variable", [])
    if not target_vars or len(target_vars) != 1:
        raise ValueError("Exactly one target variable must be specified in config['target_variable'].")

    var = target_vars[0]

    # Paths to input files
    target_path = os.path.join(config['work_dir'], "target.nc")
    driver_path = os.path.join(config['work_dir'], "driver_outputs/remote_drivers/scaled_standardized_drivers.csv")

    # Load dataset and regressors
    ds = xr.open_dataset(target_path)
    regressors = pd.read_csv(driver_path, index_col=0)
    regressors.index = regressors.index.str.strip()

    # Match models
    ds_unique = ds.groupby('model').first()
    common_models = list(regressors.index.intersection(ds_unique['model'].values))

    if not common_models:
        raise ValueError("No common models found between NetCDF and CSV data.")

    # Subset and align data
    ds_subset = ds_unique.sel(model=common_models).reindex(model=common_models)
    regressors_aligned = regressors.loc[common_models]
    regressor_names = regressors_aligned.columns.insert(0, 'MEM')

    # Extract and clean target variable
    if var not in ds_subset:
        raise ValueError(f"Variable '{var}' not found in dataset.")

    target = ds_subset[var]
    if "variable" in target.dims:
        target = target.squeeze("variable")

    # Run regression
    MLR = spatial_MLR()
    MLR.regression_data(target, regressors_aligned, regressor_names)

    # Prepare output
    output_path = os.path.join(config["work_dir"], 'regression_output')
    os.makedirs(output_path, exist_ok=True)

    output_subdir = os.path.join(output_path, var)
    os.makedirs(output_subdir, exist_ok=True)

    output_file = MLR.perform_regression(output_path, var)
    print(f"Regression completed for: {var}")

    return [output_file]

import xarray as xr
import pandas as pd
import pkg_resources

def read_data(filename):
    """
    Read a dataset from the specified filename within the package's data directory.
    
    Args:
    - filename (str): Path relative to the package's data directory.
    
    Returns:
    - xarray.Dataset: The loaded dataset.
    """
    try:
        data_path = pkg_resources.resource_filename('storypy', filename)
        data = xr.open_dataset(data_path)
        return data
    except FileNotFoundError:
        print(f"The file {filename} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Functions that utilize read_data to to load dataset

def read_pr_R2():
    return read_data('data/regression_output/pr/R2.nc')

def read_pr_regression_coefficients():
    return read_data('data/regression_output/pr/regression_coefficients.nc')

def read_pr_regression_coefficients_pvalues():
    return read_data('data/regression_output/pr/regression_coefficients_pvalues.nc')

def read_pr_regression_coefficients_relative_importance():
    return read_data('data/regression_output/pr/regression_coefficients_relative_importance.nc')


def read_ua_R2():
    return read_data('data/regression_output/ua/R2.nc')

def read_ua_regression_coefficients():
    return read_data('data/regression_output/ua/regression_coefficients.nc')

def read_ua_regression_coefficients_pvalues():
    return read_data('data/regression_output/ua/regression_coefficients_pvalues.nc')

def read_ua_regression_coefficients_relative_importance():
    return read_data('data/regression_output/ua//regression_coefficients_relative_importance.nc')


# Reading csv files
def read_drivers():
    data_path = pkg_resources.resource_filename('storypy', 'data/remote_drivers/drivers.csv')
    return pd.read_csv(data_path, index_col=0)

def read_scaled_drivers():
    data_path = pkg_resources.resource_filename('storypy', 'data/remote_drivers/scaled_drivers.csv')
    return pd.read_csv(data_path, index_col=0)

def read_scaled_standardized_drivers():
    data_path = pkg_resources.resource_filename('storypy', 'data/remote_drivers/scaled_standardized_drivers.csv')
    return pd.read_csv(data_path, index_col=0)




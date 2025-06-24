from ._regres import spatial_MLR
from storypy.utils import xr, pd
import os

def run_regression(main_config, target_var):
    """
    Run spatial multiple linear regression (MLR) using a preprocessed NetCDF dataset and regressors CSV.
    
    Parameters:
        preproc (str): Path to the preprocessed NetCDF file.
        user_config (dict): Configuration dictionary containing keys like "work_dir".
        regressor_csv_path (str): Path to the CSV file containing regressors data.
    
    This function:
      1. Opens the preprocessed NetCDF file.
      2. Loads regressors from a CSV file.
      3. Finds common models between the dataset and the regressors.
      4. Subsets the dataset based on common models.
      5. Aligns the regressors DataFrame to the common models.
      6. Prepares the regressor names by inserting 'MEM' at the beginning.
      7. Instantiates spatial_MLR, sets up regression data, and performs the regression.
      8. Saves the regression output to the specified work directory.
    """

    target_path = os.path.join(main_config['work_dir'], "combined_changes.nc")
    driver_path = os.path.join(main_config['work_dir'], "driver_test_outputs/remote_drivers/scaled_standardized_drivers.csv")
    
   
    ds = xr.open_dataset(target_path)
    # Ensure the model coordinate is a string and stripped of any whitespace.
    # ds_model_names = pd.Index(ds['model'].values.astype(str)).str.strip()
    
 
    regressors = pd.read_csv(driver_path, index_col=0)
    regressors.index = regressors.index.str.strip()  # Clean the index if necessary

    ds_unique = ds.groupby('model').first()
    common_models = list(regressors.index.intersection(ds_unique['model'].values))
    print("Common models:", common_models)
    
    # Subset and reindex using ds_unique.
    ds_subset = ds_unique.sel(model=common_models).reindex(model=common_models)
    
    # target_var = main_config(var_name)

    target = ds_subset[target_var]

    regressors_aligned = regressors.loc[common_models]

    regressor_names = regressors_aligned.columns.insert(0, 'MEM')

    # Note: spatial_MLR should be defined/imported from your module.
    MLR = spatial_MLR() # change MLR to SR
    MLR.regression_data(target, regressors_aligned, regressor_names)

    output_path = os.path.join(main_config["work_dir"], 'regression_output')
    os.makedirs(output_path, exist_ok=True)
    os.chdir(output_path)
    MLR.perform_regression(output_path, target_var)



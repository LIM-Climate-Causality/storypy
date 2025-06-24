
import os
import pandas as pd
import pandas.testing as pdt

def compare_driver_csvs(dir1, dir2, filename="drivers.csv", atol=0.01, rtol=0.01):
    """
    Compare drivers.csv files from two directories using only common models and drivers.
    
    Parameters:
        dir1 (str): Path to logic_1 output directory
        dir2 (str): Path to logic_2 output directory
        filename (str): CSV file name to compare (default: "drivers.csv")
        atol (float): Absolute tolerance
        rtol (float): Relative tolerance
    
    Returns:
        bool: True if values match within tolerance, False otherwise
    """
    file1 = os.path.join(dir1, filename)
    file2 = os.path.join(dir2, filename)

    df1 = pd.read_csv(file1, index_col=0)
    df2 = pd.read_csv(file2, index_col=0)

    # Find common models (index) and drivers (columns)
    common_models = df1.index.intersection(df2.index)
    common_columns = df1.columns.intersection(df2.columns)

    df1_common = df1.loc[common_models, common_columns].sort_index().sort_index(axis=1)
    df2_common = df2.loc[common_models, common_columns].sort_index().sort_index(axis=1)

    try:
        pd.testing.assert_frame_equal(df1_common, df2_common, atol=atol, rtol=rtol, check_dtype=False)
        print("✅ Common models and drivers match within tolerance.")
        return True
    except AssertionError as e:
        print("❌ Differences found in common model values:")
        print(e)
        return False

esmvaltool_dir = "/climca/people/ralawode/esmvaltool_output/test_recipe_20250514_132506/work/storyline_analysis/remote_drivers/remote_drivers"
directnetcdf_dir = "/climca/people/storylinetool/test_user/driver_outputs/remote_drivers"

compare_driver_csvs(esmvaltool_dir, directnetcdf_dir, filename="drivers.csv")
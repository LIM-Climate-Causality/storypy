# StoryPy
**StoryPy is an advanced toolkit that facilitates analyzing dynamical storylines by providing efficient and user-friendly tools that is flexible and adaptable for various storyline research and policy applications.**

StoryPy implements the dynamical storyline framework, presented in `Zappa&Shepherd`, using CMIP model output. It provides

- a set of functions to analyze multi-model ensembles by focusing on the identification of dynamical storylines.

- customizable options for selecting remote drivers ($X$), target seasons, and climate variables or climatic-impact drivers ($C_x$).

We designed two options for processing CMIP data:

1. Option A: Using ESMValTool (via ESMValTool recipes) to download and preprocess the CMIP datasets, including regridding.

  Requirements:
    - ESMValTool installation and a working ESMValTool environment.
    - ESMValTool recipes/configuration compatible with target variables and drivers.
    - ESMValTool preprocessing can generate large intermediate files. Ensure sufficient disk space in the working directory.

2. Option B: Using a local CMIP database where StoryPy reads CMIP-style NetCDF files directly from a local directory. Provided that:
    - CMIP datasets already available locally (or accessible via a mounted filesystem).
    - It follows the naming and grid conventions as described below.
      
    ``` <data_dir>/
        ├── <var_name>/
        │   ├── <mon>/
        │   │   ├── <g025>/
        │   │   │   ├── <var_name>_<period>_<model>_<experiment>_<member>_<grid>.nc
        │   │   │   └── ...
        │   │   └── ...
        │   └── ...
        └── ...```

Preprocessing CMIP data is a crucial step in the analysis of dynamical storylines, as it ensures that the data is in a consistent format and resolution for analysis. StoryPy provides two options for preprocessing CMIP data (as already described), either using ESMValTool or by reading from a local CMIP database. The choice between these options depends on the user's preferences and the availability of data. Given CMIP data, user can preprocess the data by calling the methods and using the following steps for example:

### Configuration and input parameter

Main user configuration

      user_config = dict(
            work_dir='/climca/people/ralawode/esmvaltool_output/zappa_shepherd_CMIP6_20260304_135924/work',
            plot_dir='/climca/people/ralawode/esmvaltool_output/zappa_shepherd_CMIP6_20260304_135924/plots',
            var_name=['pr'],
            exp_name='ssp585',
            freq='mon',
            grid='g025',
            region_method='box',
            period1 = ['1960', '1990'],
            period2 = ['2070', '2099'],
            region_id=18,
            season=(12, 1, 2),
            region_extents=[(30, 45, -10, 40), (45, 55, 5, 20)],
            # region_extents=[(25, 90, -180, 180)],
            box=([-25, 45, 30, 65]),
            titles=["Region A", "Region B"]
        )

driver configuration

      driver_config = dict(
          var_name=['psl'],            # <- actual variable names in NetCDF
          short_name=['test_ubi'],           # <- names for regression/CSV outputs
          period1=['1960', '1979'],
          period2=['2070', '2099'],
          season=[12,1,2],
          box={'lat_min': 50, 'lat_max': 70, 'lon_min': 40, 'lon_max': 70},
          work_dir='/climca/people/ralawode/esmvaltool_output/zappa_shepherd_CMIP6_20260304_135924/work'
      )

Parsing the ESMValTool configuration/metadata

    >>>from storypy.preprocess import parse_config
    >>>esmval_config= parse_config(path/settings.yml)

Running the analysis to generate climate variables $C_{xm}$ and remote drivers $X$

    >>> from storypy.preprocess import ESMValProcessor, ModelDataPreprocessor
    >>> processor_target = ESMValProcessor(esmval_config, user_config, driver_config)
    OR
    >>> processor_target = ModelDataPreprocessor(user_config, driver_config)
    >>> processor_target.process_var()
    >>> processor_target.process_driver()
    
Users can compute the driver indices and regression coefficients for a desired study region, for example:
    
For the driver indices
    
    >>> from storypy.compute import compute_drivers
    >>> df_raw, df_scaled, df_standardized = compute_drivers(driver_config)

For the regression coefficients

    >>> from storypy.compute import run_regression
    >>> outputs = run_regression(user_config)
## Installing
Storypy can be installed via PyPi. Simply run the following commands:

```pip install storypy``` or ```pip3 install storypy```

It can also be installed directly from source using pip

```git clone git@github.com:LIM-Climate-Causality/storypy.git```

```cd storypy```

```pip install .```


## If installing to be used with Esmvaltool recipe
Before installing the storypy package, users need to setup Esmvaltool on their machine. After the setup of Esmvaltool, activate the esmvaltool environment. Then, pip install the storypy package. This is because Esmvaltool uses some dependencies which are only available in conda.


Install the storypy package

```pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple storypy```



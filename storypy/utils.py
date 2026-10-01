# Aliases

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy import stats
from matplotlib import gridspec
import matplotlib as mpl
#import matplotlib.transforms as transforms
#from numpy import linalg as la

# Export these for reuse in other modules
__all__ = ['pd', 'np', 'xr', 'plt', 'mpl', 'ccrs', 'stats', 'gridspec', 'cfeature']


import os
import glob


def find_driver_csv(work_dir, explicit=None,
                    filename="scaled_standardized_drivers.csv"):
    """
    Locate the standardized driver CSV under a work directory.
    
    Parameters
    ----------
    work_dir : str
        Base directory to search under.
    explicit : str, optional
        Full path to the CSV. Returned unchanged if given and existing.
    filename : str
        File to search for. Default ``scaled_standardized_drivers.csv``.

    Returns
    -------
    str
        Path to the CSV.

    Raises
    ------
    FileNotFoundError
        If no matching file is found.
    """
    if explicit is not None:
        if not os.path.exists(explicit):
            raise FileNotFoundError(f"Driver CSV not found: {explicit}")
        return explicit

    hits = sorted(glob.glob(
        os.path.join(work_dir, "**", filename), recursive=True
    ))
    if not hits:
        raise FileNotFoundError(
            f"{filename} not found anywhere under {work_dir}. "
            "Run compute_drivers or driver_indices first, or pass the "
            "path explicitly via config['driver_csv']."
        )
    if len(hits) > 1:
        print(f"Multiple driver CSVs found under {work_dir}; using:\n  {hits[0]}")
        for h in hits[1:]:
            print(f"  (ignored) {h}")
    return hits[0]
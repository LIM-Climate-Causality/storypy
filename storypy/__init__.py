"""
storypy

A Python package to compute and visualize climate storylines.
"""

__version__ = "0.1.1"
__author__ = "Richard Alawode, Julia Mindlin, Marlene Kretschmer"
__credits__ = "LIM - Climate Causality"

# Import core functionality from the package modules
from .main import main
from .plotting import (
    plot_precipitation_change_two_regions,
    plot_function, create_three_panel_figure,
    create_five_panel_figure,
    hemispheric_plot
    )
from .processing import seasonal_data_months, apply_region_mask
from .diagnostics import clim_change, test_mean_significance
from .plot_ellipse import plot_ellipse, confidence_ellipse


#optional, we can import the commonly used libraries here
#import numpy as np
#import xarray as xr
#import matplotlib.pyplot as plt
#import cartopy.crs as ccrs
#from scipy import stats

# Explicitly define the public API of the package
__all__ = [
    "main",
    "plot_precipitation_change_two_regions",
    "plot_function",
    "seasonal_data_months",
    "apply_region_mask",
    "clim_change",
    "test_mean_significance",
    "confidence_ellipse",
    "plot_ellipse",
    "create_three_panel_figure",
    "create_five_panel_figure",
    "hemispheric_plot",
#    "np",
#    "xr",
#    "plt",
#    "ccrs",
#    "stats",
]

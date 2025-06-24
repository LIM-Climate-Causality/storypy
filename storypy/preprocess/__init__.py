from ._diagnostics import (
    extract_metadata,
    seasonal_data_months,
    clim_change,
    test_mean_significance,
    apply_region_mask,
    adjust_longitudes
)
# from ._esmval_processor import ESMValProcessor
from ._netcdf_processor import DirectProcessor



__all__ = [
    # "ESMValProcessor",
    "DirectProcessor",
    "extract_metadata",
    "seasonal_data_months",
    "clim_change",
    "test_mean_significance",
    "apply_region_mask",
    "adjust_longitudes",
]
from ._compute_driver import (
    stand_numpy,
    stand_pandas,
    compute_drivers_from_netcdf,
    driver_indices,
    _collect_scalar_drivers,
    _build_drivers_dataset,
    _select_one_file_per_var
)
from ._mlr import run_regression
from ._regres import (
    spatial_MLR,
    stand,
    stand_detr,
    filtro,
    replace_nans_with_zero,
    figure
)
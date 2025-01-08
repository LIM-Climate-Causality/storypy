"""Utilities to load sample datasets"""

# import textwrap

from ._read_data import (
    read_pr_R2,
    read_pr_regression_coefficients,
    read_pr_regression_coefficients_pvalues,
    read_pr_regression_coefficients_relative_importance,
    read_ua_R2,
    read_ua_regression_coefficients,
    read_ua_regression_coefficients_pvalues,
    read_ua_regression_coefficients_relative_importance,
    read_drivers,
    read_scaled_drivers,
    read_scaled_standardized_drivers
)

__all__ = [
    "read_pr_R2",
    "read_pr_regression_coefficients",
    "read_pr_regression_coefficients_pvalues",
    "read_pr_regression_coefficients_relative_importance",
    "read_ua_R2",
    "read_ua_regression_coefficients",
    "read_ua_regression_coefficients_pvalues",
    "read_ua_regression_coefficients_relative_importance",
    "read_drivers",
    "read_scaled_drivers",
    "read_scaled_standardized_drivers",

]
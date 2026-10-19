"""
ESMValTool is optional (conda-only, only needed for the ESMValTool workflow).
This module provides ``group_metadata`` with or without it installed.
"""

from collections import defaultdict

try:
    from esmvaltool.diag_scripts.shared import group_metadata
except ImportError:  # ESMValTool not installed: same grouping in pure Python
    def group_metadata(metadata, attribute, sort=None):
        """Group ESMValTool metadata dicts by ``attribute``."""
        groups = defaultdict(list)
        for item in metadata:
            groups[item.get(attribute)].append(item)
        if sort:
            key = sort if isinstance(sort, str) else attribute
            for k in groups:
                groups[k] = sorted(groups[k], key=lambda m: m.get(key))
        return dict(groups)

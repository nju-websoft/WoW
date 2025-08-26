"""
PyWoWLib - Python Bindings for WoWIndex
"""

# Import the C++ core extension module
# The name matches what's in setup.py: 'pywowlib._pywowlib_core'
from ._pywowlib_core import (
    _WoWIndexInt32Attr, _WoWRangeFilterInt32Attr, _WoWSetFilterInt32Attr,
    _WoWIndexInt64Attr, _WoWRangeFilterInt64Attr, _WoWSetFilterInt64Attr,
    _WoWIndexUInt32Attr, _WoWRangeFilterUInt32Attr, _WoWSetFilterUInt32Attr,
    _WoWIndexUInt64Attr, _WoWRangeFilterUInt64Attr, _WoWSetFilterUInt64Attr, _WoWBitsetLabelFilter,
    _WoWIndexLabelAttr, _WoWRangeFilterLabelAttr, _WoWSetFilterLabelAttr,  # Add LabelAttr imports
    _WoWIndexFloatAttr, _WoWRangeFilterFloatAttr, _WoWSetFilterFloatAttr,
    _WoWIndexDoubleAttr, _WoWRangeFilterDoubleAttr, _WoWSetFilterDoubleAttr,
    _FixedString16, _WoWIndexString16Attr, _WoWRangeFilterString16Attr, _WoWSetFilterString16Attr,
    _FixedString32, _WoWIndexString32Attr, _WoWRangeFilterString32Attr, _WoWSetFilterString32Attr,
)

# --- Public User-Facing Classes and Factories ---

# Expose FixedString types directly if users need to create them
FixedString16 = _FixedString16
FixedString32 = _FixedString32

# Generic filters (not dependent on att_type)
# WoWBitsetLabelFilter = _WoWBitsetLabelFilter


# --- Mappings for the factory ---
_INDEX_TYPE_MAPPING = {
    "int32": _WoWIndexInt32Attr,
    "int64": _WoWIndexInt64Attr,
    "uint32": _WoWIndexUInt32Attr,
    "uint64": _WoWIndexUInt64Attr,
    "float32": _WoWIndexFloatAttr, # Python float is C double, C float is float32
    "float": _WoWIndexFloatAttr,   # Alias for float32
    "double": _WoWIndexDoubleAttr, # Alias for float64
    "float64": _WoWIndexDoubleAttr,
    "string16": _WoWIndexString16Attr,
    "string32": _WoWIndexString32Attr,
    "label": _WoWIndexLabelAttr  # Use LabelAttr instead of UInt64Attr
}

_RANGE_FILTER_MAPPING = {
    "int32": _WoWRangeFilterInt32Attr, "int64": _WoWRangeFilterInt64Attr,
    "uint32": _WoWRangeFilterUInt32Attr, "uint64": _WoWRangeFilterUInt64Attr,
    "float32": _WoWRangeFilterFloatAttr, "float": _WoWRangeFilterFloatAttr,
    "double": _WoWRangeFilterDoubleAttr, "float64": _WoWRangeFilterDoubleAttr,
    "string16": _WoWRangeFilterString16Attr, "string32": _WoWRangeFilterString32Attr,
    "label": _WoWRangeFilterLabelAttr  # Use LabelAttr instead of UInt64Attr
}

_SET_FILTER_MAPPING = {
    "int32": _WoWSetFilterInt32Attr, "int64": _WoWSetFilterInt64Attr,
    "uint32": _WoWSetFilterUInt32Attr, "uint64": _WoWSetFilterUInt64Attr,
    "float32": _WoWSetFilterFloatAttr, "float": _WoWSetFilterFloatAttr,
    "double": _WoWSetFilterDoubleAttr, "float64": _WoWSetFilterDoubleAttr,
    "string16": _WoWSetFilterString16Attr, "string32": _WoWSetFilterString32Attr,
    "label": _WoWBitsetLabelFilter  # Keep using BitsetLabelFilter for labels
}


def WoWIndex(max_elements: int, vec_d: int, M: int, efc: int,
             space_name: str, att_type: str,
             o: int = 4, wp: int = 11, auto_raise_wp: bool = True):
    """
    Factory to create or load a WoWIndex with a specific POD attribute type.
    Supported att_type: "int32", "int64", "uint32", "uint64",
                        "float32", "float64" (or "float", "double"),
                        "string16", "string32", "label".
    """
    norm_att_type = att_type.lower().replace("_", "").replace("-", "")
    IndexClass = _INDEX_TYPE_MAPPING.get(norm_att_type)

    if IndexClass is None:
        raise ValueError(f"Unsupported att_type: '{att_type}'. Supported: {', '.join(_INDEX_TYPE_MAPPING.keys())}")

    
    return IndexClass(max_elements=max_elements, vec_d=vec_d, M=M, efc=efc,
                          space_name=space_name, o=o, wp=wp, auto_raise_wp=auto_raise_wp)
    
def WoWIndexLoad(location: str, space_name: str, att_type: str):
    """
    Factory to load a WoWIndex from a file with a specific POD attribute type.
    Supported att_type: "int32", "int64", "uint32", "uint64",
                        "float32", "float64" (or "float", "double"),
                        "string16", "string32", "label".
    """
    norm_att_type = att_type.lower().replace("_", "").replace("-", "")
    IndexClass = _INDEX_TYPE_MAPPING.get(norm_att_type)

    if IndexClass is None:
        raise ValueError(f"Unsupported att_type: '{att_type}'. Supported: {', '.join(_INDEX_TYPE_MAPPING.keys())}")

    return IndexClass(location=location, space_name=space_name)


def WoWRangeFilter(att_type: str, lower_bound, upper_bound):
    """Factory for creating a range filter based on attribute type."""
    norm_att_type = att_type.lower().replace("_", "").replace("-", "")
    FilterClass = _RANGE_FILTER_MAPPING.get(norm_att_type)
    if FilterClass is None:
        raise ValueError(f"Unsupported att_type for WoWRangeFilter: '{att_type}'.")
    return FilterClass(lower_bound=lower_bound, upper_bound=upper_bound)


def WoWSetFilter(att_type: str, *args):
    """Factory for creating a set filter based on attribute type."""
    norm_att_type = att_type.lower().replace("_", "").replace("-", "")
    FilterClass = _SET_FILTER_MAPPING.get(norm_att_type)
    if FilterClass is None:
        raise ValueError(f"Unsupported att_type for WoWSetFilter: '{att_type}'.")
    return FilterClass(*args)


__all__ = [
    "WoWIndex",
    "WoWRangeFilter",
    "WoWSetFilter",
    "WoWBitsetLabelFilter",
    "WoWBitsetIntFilter",
    "FixedString16",
    "FixedString32",
    # You can choose to also export the specific _WoWIndex... classes
    # if you want users to have access to them directly, e.g.
    # "_WoWIndexInt32Attr", "_WoWIndexString16Attr",
]
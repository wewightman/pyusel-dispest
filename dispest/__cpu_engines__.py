"""Python wrapper for c-based 1D cubic interpolators"""
import ctypes as ct
from numpy import ndarray
from glob import glob
import platform as _pltfm
import os

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# determine installed path
dirpath = os.path.dirname(__file__)

__ctype_raw__ = ct.c_float
__ctype_pnt__   = ct.POINTER(__ctype_raw__)

# determine the OS and relative binary file
if _pltfm.uname()[0] == "Windows":
    res = glob(os.path.abspath(os.path.join(dirpath, "*.dll")))
    name = res[0]
elif _pltfm.uname()[0] == "Linux":
    res = glob(os.path.abspath(os.path.join(dirpath, "*.so")))
    name = res[0]
else:
    res = glob(os.path.abspath(os.path.join(dirpath, "*.dylib")))
    name = res[0]

# load the c library
__cpu_dispest__ = ct.CDLL(name)

__cpu_dispest__.calc_nxc_and_std_lagpairs.argtypes = (
    ct.c_longlong,
    ct.c_longlong,
    ct.POINTER(ct.c_longlong),
    ct.c_longlong,
    ct.c_longlong,
    ct.POINTER(ct.c_float),
    ct.POINTER(ct.c_float),
    ct.POINTER(ct.c_float),
    ct.POINTER(ct.c_float),
    ct.POINTER(ct.c_float),
    ct.POINTER(ct.c_float),
    ct.c_float
)
__cpu_dispest__.calc_nxc_and_std_lagpairs.restype = (None)
__cpu_dispest__.calc_nxc_and_std_lagpairs.__doc__ = """Calculate the normalized cross correlation for a given signal pair and save signal STD in the matched reference and search kernels"""

__cpu_dispest__.calc_nxc_lagpairs.argtypes = (
    ct.c_longlong,
    ct.c_longlong,
    ct.POINTER(ct.c_longlong),
    ct.c_longlong,
    ct.c_longlong,
    ct.POINTER(ct.c_float),
    ct.POINTER(ct.c_float),
    ct.POINTER(ct.c_float),
    ct.POINTER(ct.c_float),
    ct.c_float
)
__cpu_dispest__.calc_nxc_lagpairs.restype = (None)
__cpu_dispest__.calc_nxc_lagpairs.__doc__ = """Calculate the normalized cross correlation for a given signal pair"""


def __offset_pnt__(pnt, offset:int):
    """Helper function to offset a ctypes pointer by a given numbe of elements"""
    from ctypes import sizeof, cast, POINTER, c_void_p

    # convert to a void-type pointer object
    pvoid = cast(pnt, c_void_p)

    # calculate the element offset in bytes
    pvoid.value += int(offset * sizeof(pnt._type_))

    # reform the pointer as its original type
    pnt_out = cast(pvoid, POINTER(pnt._type_))

    return pnt_out

def __get_contig_pointer__(arr:ndarray, ctype):
    """Convert a numpy array into a ctypes pointer"""
    import numpy as np
    import ctypes as ct
    arr_cont = np.ascontiguousarray(arr, dtype=ctype)
    arr_pnt  = arr_cont.ctypes.data_as(ct.POINTER(ctype))

    return arr_pnt

def __get_pointer__(arr:ndarray, ctype):
    """Convert a numpy array into a ctypes pointer"""
    import numpy as np
    arr_pnt  = arr.ctypes.data_as(ct.POINTER(ctype))

    return arr_pnt
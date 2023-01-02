from typing import List

import numpy as np
from numba import jit, types

jit_settings = {'nopython': True, 'nogil': True, 'cache': True}

fun_signature = "void(float64[::1], float64 , float64[::1], float64[::1])"
dfun_signature = "FunctionType(" + fun_signature + ")"
vfloat_sig = types.Array(dtype=types.float64, ndim=1, layout="C")

@jit("float64[:, ::1](ListType(float64[::1]))", **jit_settings)
def asmatrix(list: List) -> np.ndarray:
    """Convert List to numpy C-contiguous array.

    This method is used to overload numpy built-in asarray() and asanyarray() methods
    which are currently not supported by numba.

    Parameters
    ----------
    list : List
        List to be converted.

    Returns
    -------
    output : ndarray, shape (n_elements, ndof)
        Converted list
    """

    sy, sx = len(list[0]), len(list)
    y = np.zeros((sx,sy))
    for i in range(sx):
        y[i] = list[i]

    return y


@jit("b1(float64, float64, float64, float64)", **jit_settings)
def isclose(x: float, y: float, atol: float, rtol: float) -> bool:
    """Check whether two scalars are equal within the given tolerances.

    Parameters
    ----------
    x, y : float
        Values that must be compared.
    atol, rtol : float
        Absolute and relative desired comparison tolerances.

    Returns
    -------
    output : boolean
        Comparison result
    """
    return abs(y-x) <= atol + abs(x)*rtol

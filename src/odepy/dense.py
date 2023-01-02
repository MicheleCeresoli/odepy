from typing import Union, Callable

import numpy as np
from numba import jit

from odepy.utils import dfun_signature, jit_settings

@jit("float64[:, ::1](" + dfun_signature + ", float64, float64[::1], float64[::1], "
     "float64[::1], float64[::1], float64, float64[:, ::1], float64[:, ::1],"
     "float64[::1], float64[:, ::1], int64, int64, b1)",
     **jit_settings)
def dense_output(fun: Callable[[np.ndarray, float, np.ndarray, np.ndarray], None], t: float,
                 y: np.ndarray, args: np.ndarray, y_new: np.ndarray, f: np.ndarray,
                 h: float, K: np.ndarray, A: np.ndarray, C: np.ndarray, P: np.ndarray,
                 stages: int, extra_stages: int, DOP: bool) -> np.ndarray:
    """Compute the dense-output coefficients.

    Parameters
    ----------
    fun : callable(ndarray, float, ndarray, ndarray)
        Dynamic function to integrate.
    t : float
        Current time.
    y, y_new : ndarray, shape (ndof,)
        Current and next states.
    args : ndarray, shape (nargs,)
        Dynamic function parameters.
    f: ndarray, shape (ndof,)
        Current derivative value, i.e., ``fun(t, y)``.
    h : float
        Integration step.
    K : ndarray, shape (stages, ndof)
        Storage array containing the RK stages. Each stage is stored in a separate row,
        so that the last row is a linear combination of all the previous rows.
    A : ndarray, shape (stages + extra-stages, stages + extra-stages)
        Runge-Kutta square matrix. The matrix includes the
        extra-coefficients required for dense-outputs.
    C : ndarray, shape (stages + extra-stages,)
        Runge-Kutta nodes vector. The vector includes the nodes required for dense-outputs.
    P : ndarray, shape (interpolation-order, stages+extra-stages)
        Dense-output coefficients.
    stages, extra_stages : int
        Number of stages and extra-stages.
    DOP : boolean
        True if the solver is DOP853.

    Returns
    -------
    Q : ndarray, shape (interpolation-order, ndof)
        Interpolating polynomial coefficients.

    Notes
    -----
    The function implements a custom routine to compute the polynomial coefficients
    for DOP853.

    References
    ----------
    .. [1] Scipy DOP853 Implementation.
    .. [2] RKSuite.cpp codes: <http://www.netlib.org/ode/rksuite/>.
    """

    ndof = len(y)

    if extra_stages > 0:
        # Computes the remaining extra-stages-th function evaluations
        K_aug = np.zeros((stages+extra_stages, ndof))
        K_aug[:stages] = K

        ti = t + h*C[stages:]

        for i in range(stages, stages+extra_stages):
            yi = y + h*np.dot(A[i, :i], K_aug[:i])
            fun(K_aug[i], ti[i-stages], yi, args)

        # RKSUITE.cpp takes the summation (instead of dot product between P and K)
        # because it minimises round-off errors, however its implementation here
        # would require a dedicated summation for each interpolant and solver
        Q = P.dot(K_aug)
    else:
        Q = P.dot(K)

    # Custom dense-output matrix for DOP853 (see SciPy):
    if DOP:
        dy = y_new - y

        F = np.empty((7, ndof))
        F[0] = dy
        F[1] = h*K[0] - dy
        F[2] = 2*dy - h*(f + K[0])
        F[3:] = h*Q

        Q = F

    return Q

@jit(["float64[:, ::1](float64[::1], float64[::1], float64, float64, float64[:, ::1], b1)",
      "float64[:, ::1](float64, float64[::1], float64, float64, float64[:, ::1], b1)"],
     **jit_settings)
def interpolate(tms: Union[float, np.ndarray], yk: np.ndarray, tk: float,
                hk: float, Q: np.ndarray, DOP: bool) -> np.ndarray:
    """Interpolate the solution at the desired times.

    The evaluation of the interpolating polynomial at tms is done
    with Horner's rule: i.e. `f(x) = a + bx + cx^2 + dx^3` can be
    written and evaluated as: `f(x) = a + x(b + x(c + xd))`.
    The advantage of offered by this form is that it reduces the number
    of multiplication operations. Since the processing time of a single
    multiplicatiion is typically 5 to 20 times the processing time of an
    addition, this form is much faster than the original Scipy's version.

    Parameters
    ----------
    tms : ndarray, shape (npoints,) or float
        Interpolation points. Its elements do not have to follow any
        particular ordering.
    yk : ndarray, shape (ndof,)
        Current solution.
    tk : float
        Current time.
    hk : float
        Integration step.
    Q : ndarray, shape (interpolation-order, ndof)
        Interpolating polynomial coefficients.
    DOP : boolean
        True if the solver is DOP853.

    Notes
    -----
    This function does not perform any checks on the integration points.
    However, if a point it outside [tk, tk+hk] the accuracy of the solution is
    not guaranteeed.

    Returns
    -------
    y : ndarray, shape (npoints, ndof)
        Interpolated solution.

    References
    ----------
    .. [1] Scipy's DOP853 Implementation

    """

    # Creates an array if tms is a scalar float
    tms = np.atleast_1d(np.asarray(tms))

    # Scaled time
    x = np.expand_dims((tms - tk)/hk, axis=1)

    if not DOP:
        y = Q[-1]*x
        for k in range(Q.shape[0]-2, -1, -1):
            y += Q[k]
            y *= x

        return hk*y + yk

    else: # Custom DOP implementation (see Scipy codes)
        y = np.zeros((len(x), len(yk)))
        for k in range(Q.shape[0]-1, -1, -1):
            y += Q[k]
            if k % 2 == 0:
                y *= x
            else:
                y *= (1-x)

        return y + yk

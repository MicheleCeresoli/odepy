from typing import Tuple

import numpy as np
from numba import jit

from odepy.utils import jit_settings

SAFETY = 0.9
MIN_FACTOR = 0.2
MAX_FACTOR = 10

controller_signature = "float64[::1]()"

@jit("Tuple((float64, b1))(float64[::1], float64[::1], b1)",
     **jit_settings)
def _basic_control(errs: np.ndarray, betas: np.ndarray,
                   step_rejected: bool) -> Tuple[float, bool]:
    """Compute the step-size multiplier for integral and PI controllers.

    Parameters
    ----------
    errs : ndarray, shape (3,)
        Error norms of the current and previous two successful steps.
    betas : ndarray, shape (3,)
        Step-size controller parameters.
    step_rejected : boolean
        True if a previous iteration of the same step (i.e., with the same tk, yk)
        has already been rejected.

    Returns
    -------
    factor : float
        Step-size multiplier.
    step_rejected : boolean
        True if the current step must be rejected.

    References
    ----------
    .. [1] DifferentialEquations.jl documentation:
        <https://diffeq.sciml.ai/stable/extras/timestepping/>
    .. [2] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
           Equations I: Nonstiff Problems", Section II.4
    .. [3] G. Söderlind, L. Wang "Adaptive Time-Stepping and Computational Stability"
    .. [4] G. Söderlind, "Digital Filters in Adaptive Time-Stepping"
    """

    if errs[-2] == -1: # Integral Controller
        factor = errs[-1]**betas[0]
    else: # PI Controller
        factor = errs[-1]**betas[0]*errs[-2]**betas[1]

    factor *= SAFETY
    if errs[-1] < 1:
        factor = min(MAX_FACTOR, factor)
        if step_rejected:
            factor = min(1, factor)

        return factor, False
    else:
        factor = max(MIN_FACTOR, factor)
        return factor, True


@jit("Tuple((float64, b1))(float64[::1], float64[::1])",
     **jit_settings)
def _pid_control(errs: np.ndarray, betas: np.ndarray) -> Tuple[float, bool]:
    """Compute the step-size multiplier for a PID controller.
 nopython=True, nogil=True, cache=True
    Differently from the integral and PI controllers, for a PID a limiter
    function is used to replace the factor clipping (between a min and max
    value) aswell as to account for the safety factor. Similarly to DE.jl, the
    step is rejected whenever the proposed step-multiplier is bigger than a given
    threshold (0.82). If the step is rejected, the step is re-tried with this newly
    predicted step-size.

    Parameters
    ---------
    errs : ndarray, shape (3,)
        Error norms of the current and previous two successful steps.
    betas : ndarray, shape (3,)
        PID controller parameters.

    Returns
    -------
    factor : float
        Step-size multiplier.
    step_rejected : boolean
        True if the current step must be rejected.

    References
    ----------
    .. [1] DifferentialEquations.jl documentation:
        <https://diffeq.sciml.ai/stable/extras/timestepping/>
    .. [2] G. Söderlind, L. Wang "Adaptive Time-Stepping and Computational Stability"
    .. [3] G. Söderlind, "Digital Filters in Adaptive Time-Stepping"
    """


    factor = errs[-1]**betas[0]*errs[-2]**betas[1]*errs[0]**betas[2]
    factor = 1 + np.arctan(factor - 1)

    if factor >= 0.82:
        return factor, False
    else:
        return factor, True


@jit("Tuple((float64, b1))(float64[::1], float64[::1], b1)",
     **jit_settings)
def stepsize_controller(errs: np.ndarray, betas: np.ndarray,
                        step_rejected: bool) -> Tuple[float, bool]:
    """Compute step-size adaptive factor.

    This function automatically calls the correct step-size control depending
    on the controller type (i.e., the value of beta[-1]). Whenever the current
    error norm is null, MAX_FACTOR is returned irrespective of the controller type
    to avoid numerical issues.

    Parameters
    ----------
    errs : ndarray, shape (3,)
        Error norms of the current and previous two successful steps.
    betas : ndarray, shape (3,)
        Step-size controller parameters.
    step_rejected : boolean
        True if a previous iteration of the same step (i.e., with the same tk, yk)
        has already been rejected.

    Returns
    -------
    factor : float
        Step-size multiplier.
    rejected : boolean
        True if the current step is rejected.
    """

    if errs[-1] == 0:
        factor = MAX_FACTOR
        return factor, False

    if betas[-1] == 0 or errs[0] < 0: #I\PI Controllers
        return _basic_control(errs, betas, step_rejected)
    else:
        return _pid_control(errs, betas)


@jit(controller_signature, **jit_settings)
def integral_controller() -> Tuple[float, float, float]:
    """Return integral controller parameters.

    Returns
    -------
    betas : ndarray, shape (3,)
    """

    return np.array([1., 0., 0.])


@jit(controller_signature, **jit_settings)
def PI42() -> Tuple[float, float, float]:
    """Return PI42 controller parameters.

    Returns
    -------
    betas : ndarray, shape (3,)
    """

    return np.array([0.6, -0.2, 0.])


@jit(controller_signature, **jit_settings)
def PI33() -> Tuple[float, float, float]:
    """Return PI33 controller parameters.

    Returns
    -------
    betas : ndarray, shape (3,)
    """

    return np.array([2./3, -1./3, 0.])


@jit(controller_signature, **jit_settings)
def PI34() -> Tuple[float, float, float]:
    """Return PI34 controller parameters.

    Returns
    -------
    betas : ndarray, shape (3,)
    """

    return np.array([0.7, -0.4, 0.])


@jit(controller_signature, **jit_settings)
def H211PI() -> Tuple[float, float, float]:
    """Return H211PI controller parameters.

    Returns
    -------
    betas : ndarray, shape (3,)
    """

    return np.array([1./6, 1./6, 0.])


@jit(controller_signature, **jit_settings)
def H312PID() -> Tuple[float, float, float]:
    """Return H312PID controller parameters.

    Returns
    -------
    betas : ndarray, shape (3,)
    """

    return np.array([1./18, 1./9, 1./18])


from typing import Callable, Tuple

import numpy as np
from numba import jit

from odepy.controllers import stepsize_controller
from odepy.tableaus import rk_method_signature as rkms
from odepy.utils import dfun_signature, jit_settings

@jit("float64(" + dfun_signature + ", float64[::1], float64[::1], float64[::1],"
     "float64[::1], int64, float64, float64, int64)",
     **jit_settings)
def compute_initial_step(fun: Callable[[np.ndarray, float, np.ndarray, np.ndarray], None],
                          t_bounds: np.ndarray, y0: np.ndarray, args: np.ndarray,
                          f0: np.ndarray, ndof: int, atol: float, rtol: float,
                          error_estimation_order: int) -> float:
    """Compute the initial integration step size.

    This function automatically computes a starting step size using
    estimates of the local Lipschitz constant to compute a bound for the
    second derivative of the input function [1].

    Parameters
    ----------
    fun : callable (ndarray, float, ndarray, ndarray)
        Dynamic function to integrate.
    t_bounds: ndarray, shape (2,)
        Integration time interval.
    y0 : ndarray, shape (ndof,)
        Initial state.
    args : ndarray, shape (nargs,)
        Dynamic function parameters.
    f0 : ndarray, shape (ndof,)
        Initial derivative value, i.e, ``fun(t0, y0)``.
    atol, rtol : float
        Relative and absolute tolerances.
    error_estimation_order: int
        Estimation order for the local truncation error.

    Results
    -------
    absh : float
        Initial step size.

    References
    ----------
    [1] Watts, H. A. "Starting step size for an ODE solver."
    """

    big = np.sqrt(np.finfo(y0.dtype).max)
    small = np.nextafter(np.finfo(y0.dtype).epsneg, 1.0)  # paper version of u

    # Defines interval parameters (b is selected as final point)
    a, b = t_bounds[0], t_bounds[-1]

    dx = b - a
    dx_abs = abs(dx)
    dx_sgn = np.sign(b-a)

    small_pow = small**0.375

    # Computes the time-perturbation magnitude da:
    da = max(min(small_pow*abs(a), dx_abs), 100*small*abs(a))*dx_sgn
    if da == 0.0:  # Handles when da underflows to zero
        da = small_pow*dx

    # 1. COMPUTES DFDX
    fda = np.empty_like(y0)
    fun(fda, a + da, y0, args)

    dfdx_norm = np.linalg.norm(fda - f0)/abs(da)
    dfdx_norm = min(dfdx_norm, big)  # Avoids overflow

    y0_norm = np.linalg.norm(y0)
    f0_norm = np.linalg.norm(f0)

    # 2. SEARCH FOR LOCAL LIPSHITZ CONSTANT
    # Computes initial perturbation vector
    dy_norm = small_pow
    if y0_norm:  # Handles underflow if norm(y0) == 0
        dy_norm *= y0_norm

    # At the first iteration non-zero components are not removed.
    if f0_norm:
        dy_i = dx_sgn*dy_norm/f0_norm*f0
    else:
        e = np.ones_like(y0)
        dy_i = dx_sgn*dy_norm*e

    lipsch = 0.0
    fi = np.empty_like(y0)
    for j in range(1, max(ndof+2, 4)):  # Compie massimo 3 iterazioni
        yi = y0 + dy_i

        if not j == 2:
            fun(fi, a, yi, args)
            fo = f0

            if j == 1:
                yi_sgn = np.sign(dy_i)
                yi_sgn = np.where(yi_sgn, yi_sgn, np.sign(dx*fi))
                yi_sgn[yi_sgn == 0] = np.sign(dx)

        else:
            fun(fi, a + da, yi, args)
            fo = fda

        lipsch = max(lipsch, np.linalg.norm(fi - fo)/dy_norm)

        # Computes new perturbation vector
        if j == 1:
            dy_i = yi_sgn*np.linalg.norm(fi - fo)/lipsch
        else:
            if y0_norm == 0.0:
                wk = np.ones_like(y0)
            else:
                wk = np.copy(y0)
                wk[wk == 0.0] = y0_norm

            dy_i = yi_sgn*dy_norm/np.linalg.norm(wk)*wk

    # Computes the upper bound for the second derivative of y
    ddy_norm = dfdx_norm + lipsch*f0_norm

    tol = atol + rtol*np.abs(y0)  # tau: user-defined tolerance
    tol_log = np.log10(tol)

    q = 0.5*(np.sum(tol_log)/ndof + np.min(tol_log))
    eps = 10**(q/(error_estimation_order+1))

    if ddy_norm:
        absh = eps/np.sqrt(0.5*ddy_norm)
    elif f0_norm:   # if only the second derivative is null
        absh = eps/f0_norm
    else:           # if both first and second derivative are null
        absh = dx_abs*eps

    if lipsch:  # avoids division by 0
        absh = min(absh, 1/lipsch)

    absh = min(dx_abs, max(absh, 100*small*abs(a)))
    if absh == 0.0:
        absh = small*abs(b)

    return absh


@jit("float64(float64[::1], float64[::1], float64, float64,"
     "float64, float64[:, ::1], float64[::1], float64[::1])",
     **jit_settings)
def _estimate_error_norm(y: np.ndarray, y_new: np.ndarray, absh: float, atol: float,
                         rtol: float, K: np.ndarray, E: np.ndarray, E3: np.ndarray) -> float:
    """Estimate local Runge-Kutta truncation error.

    This function computes a scaled error estimate on each component
    of `y`. The output is the Hairer norm of this scaled error estimate [1].
    Its advantage is that it does not change if new equations are added to the model:
    the error is scaled by the the number of equations so that independent equations
    will not step differently than a single solve [2]. A result greater than one means
    the error does not satisfy the tolerances.

    Parameters
    ----------
    y, y_new : ndarray, shape (ndof,)
        Previous and newly computed states.
    absh : float
        Current absolute step size.
    atol, rtol: float
        Relative and absolute tolerances. The solver keeps the local error estimates
        smaller than ``atol + rtol * abs(y)``. `atol` controls the absolute accuracy
        (number of corrected decimal places), whereas `rtol` controls the relative
        accuracy (number of correct digits). Note that this tolerances are local
        tolerances and thus are not global guarantees.
    K : ndarray, shape (stages, ndof)
        Storage array containing the RK stages. Each stage is stored in a separate row,
        so that the last row is a linear combination of all the previous rows.
    E : ndarray, shape (stages,)
        Coefficients for the local error estimation
    E3: ndarray, shape (stages,)
        Coeffficients for the 3rd order local error estimation.

    Returns
    -------
    error: float
        Estimate of the local Runge-Kutta truncation error.

    Notes
    -----
    This function implements the custom DOP853 error estimation depending on the
    value of the first element of E3. If that does not match the DOP853 signature,
    the traditional formula is used [1][3][4].

    References
    ----------
    .. [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
           Equations I: Nonstiff Problems"
    .. [2] DifferentialEquations.jl documentation:
        <https://diffeq.sciml.ai/stable/extras/timestepping/#timestepping>
    .. [3] G. Wanner, E. Hairer, "Solving Ordinary Differential Equations II, Section IV.2"
    .. [4] Scipy DOP853 Implementation.

    """

    scale = atol + np.maximum(np.abs(y), np.abs(y_new))*rtol
    # scale = atol + 0.5 * (np.abs(y) + np.abs(y_new)) * rtol

    if E3[0] == 0:
        y_err = absh*np.dot(E, K)
        return np.sqrt(np.sum((y_err/scale)**2)/len(y))  # Hairer norm
        # return np.linalg.norm(absh * y_err / scale)

    else:  # DOP853 Custom Error Estimation
        err5 = np.dot(E, K) / scale
        err3 = np.dot(E3, K) / scale

        err5_norm = np.linalg.norm(err5) ** 2
        err3_norm = np.linalg.norm(err3) ** 2

        if err5_norm == 0 and err3_norm == 0:
            return 0.0
        else:
            denom = err5_norm + 0.01 * err3_norm
            return absh * err5_norm / np.sqrt(denom * len(scale))


@jit("float64[::1](" + dfun_signature + ", float64, float64[::1], float64[::1],"
     "float64[::1], float64, float64[:, ::1], float64[::1], float64[::1], float64[:, ::1],"
     "int64, b1)", **jit_settings)
def _base_rk_step(fun: Callable[[np.ndarray, float, np.ndarray, np.ndarray], None], t: float,
                  y: np.ndarray, args: np.ndarray, f: np.ndarray, h: float, A: np.ndarray,
                  B: np.ndarray, C: np.ndarray, K: np.ndarray, stages: int,
                  FSAL: bool) -> Tuple[float, np.ndarray]:
    """Compute a single Runge-Kutta step.

    Parameters
    ----------
    fun : callable(ndarray, float, ndarray, ndarray)
        Dynamic function to integrate.
    t : float
        Current time.
    y : ndarray, shape (ndof,)
        Current state.
    args : ndarray, shape (nargs,)
        Dynamic function parameters.
    f : ndarray, shape (ndof,)
        Current derivative value, i.e, ``fun(t, y)``.
    h : float
        Integration step.
    A : ndarray, shape (stages + extra-stages, stages + extra-stages)
        Runge-Kutta square matrix. The matrix includes the
        extra-coefficients required for dense-outputs.
    B : ndarray, shape (stages,)
        Runge-Kutta weights vector. The vector is empty because DOP853 is a FSAL method.
    C : ndarray, shape (stages + extra-stages,)
        Runge-Kutta nodes vector. The vector includes the nodes required for dense-outputs.
    K : ndarray, shape (stages, ndof)
        Storage array containing the RK stages. Each stage is stored in a separate row,
        so that the last row is a linear combination of all the previous rows.
    stages: int
        Number of RK stages.
    FSAL: bool
        True if the integration method is FSAL (First Same As Last).

    Returns
    -------
    y_new : ndarray, shape (ndof,)
        Solution at t + h (t_new)
    """

    ti = t + h * C[:stages]

    # This copy is a must, otherwise when f is modified at the end of the step, K will
    # change aswell, despite the original K might be needed for the dense-output evaluation.
    K[0] = np.copy(f)
    for i in range(1, stages):
        yi = y + h * np.dot(A[i, :i], K[:i])
        fun(K[i], ti[i], yi, args)

    if FSAL:
        y_new = yi
    else:
        y_new = y + h * np.dot(B, K)

    return y_new


@jit("Tuple((float64, float64[::1], float64, int64))(" + dfun_signature + ", float64,"
     "float64[::1], float64[::1], float64[::1], float64, float64, float64, float64,"
     "float64, float64[:, ::1], float64, float64[::1], float64[::1],"
     + rkms[:-2] + ")", **jit_settings)
def perform_step(fun: Callable[[np.ndarray, float, np.ndarray, np.ndarray], None], t: float,
                  y: np.ndarray, args: np.ndarray, f: np.ndarray, absh: float, t_dir: float,
                  t_end: float, atol: float, rtol: float, K: np.ndarray,
                  hmin: float, errs: np.ndarray, betas: np.ndarray,
                  params: Tuple) -> Tuple[float, np.ndarray, float, int]:
    """Advance the Runge-Kutta integration of a step.

    This function iterates and adapts the integration step until
    the step is accepted by the desired step-size controller. It
    raises a `RuntimeError` if the step-size has to be reduced below
    the minimum allowed value, which depends on the machine precision.

    Parameters
    ----------
    fun : Callable(ndarray, float, ndarray, ndarray)
        Dynamic function to integrate.
    t : float
        Current time
    y : ndarrray, shape (ndof,)
        Current state.
    args : ndarray, shape (nargs,)
        Dynamic function parameters
    f : ndarray, shape (ndof,)
        Current derivative value, i.e, ``fun(t, y)``.
    absh : float
        Current absolute integration step.
    t_dir : float
        Integration direction
    t_end : float
        Final integration time
    atol, rtol : float
        Absolute and relative integration tolerances.
    K : ndarray, shape ()
        Storage array containing the RK stages. Each stage is stored in a separate row,
        so that the last row is a linear combination of all the previous rows.
    hmin : float
        Minimum allowed step-size.
    errs : ndarray, shape (3,)
        Error norms of the current and previous two successful steps.
    betas : ndarray, shape (3,)
        Step-size controller parameters.
    params : Tuple
        RK solver parameters as returned by the tableau functions.

    Returns
    -------
    t_new : float
        New integration time.
    y_new : ndarray, shape (ndof,)
        New integration state.
    absh : float
        Adapted integration step-size for the next step.
    e_flag : int
        Exit flag:
         * 0: The step was successful.
         * 1: The step failed because the required step-size
              is below the minimum allowed size.
    Notes
    -----
    This function uses in-place operations to update `f` with the
    value computed at `f(t_new, y_new, args)` and the `errs` vector.
    """

    e_flag = 0
    t_new = t

    step_rejected, step_accepted = False, False
    while not step_accepted:

        if absh < hmin:
            e_flag = 1
            break

        absh = min(absh, abs(t_end - t))

        y_new = _base_rk_step(fun, t, y, args, f, absh*t_dir, params[0], params[1],
                              params[2], K, params[9], params[6])

        errs[-1] = _estimate_error_norm(y, y_new,
                                        absh, atol, rtol, K, params[3], params[4])

        factor, step_rejected = stepsize_controller(errs, betas, step_rejected)

        if not step_rejected:
            t_new += t_dir*absh
            step_accepted = True

            # Updates errors for controllers
            errs[:2] = errs[1:]

        absh *= factor

    # Once the step is completed, the step-vector is updated:
    if params[6]:  # FSAL
        # Copy is necessary because if K is modified and the step is rejected,
        # the previous f is not available anymore.
        f[:] = np.copy(K[params[9]-1])
    else:
        fun(f, t_new, y_new, args)

    return t_new, y_new, absh, e_flag

from typing import Callable, Tuple, List

import numpy as np
from numba import jit, types
from numba.typed import List as nbList

from odepy.dense import interpolate
from odepy.utils import vfloat_sig, jit_settings

event_signature = "Tuple((float64, int64, int64))(float64, float64[::1], float64[::1])"
_event_fcn_signature = "FunctionType(" + event_signature + ")"
event_wrapper_signature = "ListType(" + _event_fcn_signature + ")"

_event_list_signature = types.FunctionType(
    types.Tuple((types.float64, types.int64, types.int64))(
        types.float64, vfloat_sig, vfloat_sig))

@jit(nopython=True, nogil=True)
def assemble_events(
        events: Tuple[Callable[[float, np.ndarray, np.ndarray],
                               Tuple[float, int, int]]]
        ) -> List[Tuple[Callable[[float, np.ndarray, np.ndarray],
                                 Tuple[float, int, int]]]]:

    """Creates a typed-list of event functions for the integration.

    Parameters
    ----------
    events : Tuple, shape(n_events, )
        A tuple of jitted event functions. Each function shall have the
        signature: ``event(t, y, args)`` and return a tuple of three arguments:

            value : float
                The result of an expression which describes the event.
                The event happens when value is equal to zero.
            isterminal : int
                Specifies whether the integration shall be stopped if
                this specific event occurs. A value of 1 will terminate
                the integration on occurance.
            direction : int
                Direction of a zero crossing. If direction = 0 all zeros
                will be located. A positive `direction` will trigger the
                event only when the event `value` goes from negative to
                positive and viceversa if `direction` is negative.

    Returns
    -------
    event_list : Typed List, shape (n_events,)
        A numba typed-list containing the input event functions.
    """

    event_list = nbList.empty_list(_event_list_signature)

    for e in events:
        event_list.append(e)

    return event_list


@jit("int64[::1](float64[::1], float64[::1], int64[::1])",
     **jit_settings)
def get_active_events(g: np.ndarray, g_new: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    """Return the active events indexes.

    Parameters
    ----------
    g, g_new : ndarray, shape (n_events,)
        Previous and current event values.
    dirs : ndarray, shape (n_event,)
        Event direction that accounts for the increasing
        or decreasing behaviour of the event function.

    Returns
    -------
    output : ndarray, shape (n_active_events, )
        Array containing the indexes of the active events.
    """

    up = (g <= 0) & (g_new >= 0)
    down = (g >= 0) & (g_new <= 0)
    either = up | down
    mask = (up & (dirs > 0) |
            down & (dirs < 0) |
            either & (dirs == 0))

    return np.nonzero(mask)[0]


@jit("Tuple((b1, float64))(" + _event_fcn_signature + ", float64[::1]," +
     "float64[::1], float64[:, ::1], b1, float64, float64, float64, int64)",
     **jit_settings)
def ridder(event: Callable[[float, np.ndarray, np.ndarray], Tuple[float, int, int]],
           yk: np.ndarray, args: np.ndarray, Q: np.ndarray, DOP: bool, tk: float,
           tkp: float, atol: float, maxiter: int) -> Tuple[bool, float]:
    """
    Find a root of an event function within an interval using Ridder's method.

    A dedicated event function is declared within this method because Numba does not
    allow lambda functions to be passed between jitted-functions. This function
    transforms the input event `event(t, y, args)` into `event(t, y(t), args)`,
    with `y(t)` is the dense-output solution between `tk` and `tkp`.

    Parameters
    ----------
    event : Callable
        Jitted event function returning a number and taking a scalar variable
        as input (i.e. time for example).
        f must be continuous, and f(a) and f(b) must have opposite signs.
    yk : ndarray, shape (ndof,)
        Current integration state.
    args : ndarray, shape (n, )
        Event function parameters.
    Q : ndarray, shape (interpolation-order, ndof)
        Interpolating polynomial coefficients.
    DOP : boolean
        True if the solver is DOP853.
    tk, tkp : float
        Current and next integration times.
    atol : float
        The computed root ``t0`` will satisfy ``np.allclose(t, t0,
        atol=atol)``, where ``t`` is the exact root. The
        parameter must be nonnegative.
    maxiter : int, optional
        If convergence is not achieved in `maxiter` iterations, an error is
        raised. Must be >= 0, by default 50.

    Returns
    -------
    converged: bool
        A boolean reporting the convergence of the method or not.
    t0 : Union[np.ndarray, float]
        Zero of `f` between `a` and `b`.

    """

    def fun(t):
        return event(t, interpolate(t, yk, tk, tkp-tk, Q, DOP)[0], args)[0]

    # Ensures backward propagation also works!
    if tk < tkp:
        a, b = tk, tkp
    else:
        a, b = tkp, tk

    fa = fun(a)
    if fa == 0.0:
        return True, a

    fb = fun(b)
    if fb == 0.0:
        return True, b

    if fa*fb > 0: # slightly faster than comparing the signs
        raise Exception('Root is not bracketed!')

    tOld = a

    for i in range(0, maxiter):
        # Compute the improved root x from Ridder's formula
        c = 0.5*(a + b)
        fc = fun(c)

        if fc == 0:
            return True, c
        else:
            s = np.sqrt(fc**2 - fa*fb)

            if s == 0.0:
                return False, c

            dt = (c - a)*fc/s

            if (fa - fb) < 0.0:
                dt = -dt

            t = c + dt
            ft = fun(t)

            # Test for convergence
            if abs(ft) == 0.0 or (i > 0 and abs(t-tOld) < atol):
                return True, t

            tOld = t

            # Re-bracket the root
            if fc*ft > 0:
                if fa*ft < 0:
                    b = t
                    fb = ft
                else:
                    a = t
                    fa = ft
            else:
                a = c
                b = t
                fa = fc
                fb = ft

    return False, t

@jit("Tuple((int64[::1], float64[::1], b1, b1))(" + event_wrapper_signature +
     ", int64[::1], int64[::1], float64[::1], float64[::1], float64[:, ::1]," +
     "b1, float64, float64, float64, int64)", **jit_settings)
def handle_events(events: List[Callable[[float, np.ndarray, np.ndarray],
                                        Tuple[float, int, int]]],
                  active_events: np.ndarray, is_terminal: np.ndarray,
                  yk: np.ndarray, args: np.ndarray, Q: np.ndarray, DOP: bool,
                  tk_old: float, tk: float, event_tol: float, maxiter: int):
    """Compute the root of any active event within a given interval.

    This function computes the roots of all active events within the
    given integration interval. If a terminal event is trigged at ``t_1``,
    only the roots associated to the events that happen before ``t_1``.
    are returned.

    Parameters
    ----------
    events : TypedList(Callables)
        Typed list of all event functions.
    active_events : ndarray, shape (nactive, )
        Array containing the indexes of each active event
        between [tk_old, tk] or [tk, tk_old] depending on the
        integration direction.
    is_terminal : ndarray, shape (nevents, )
        Array that specifies whether each event in `events`
        is terminal or not.
    yk : ndarray, shape (ndof, )
        Current integration state.
    args : ndarray, shape (nargs, )
        Dynamic and event function arguments
    Q : ndarray, shape (interpolation-order, ndof)
        Interpolating polynomial coefficients.
    DOP : boolean
        True if DOP853 is the selected solver
    tk_old, tk : float
        Current and new integration times.
    event_tol : float
        Absolute tolerance for the computed event root.
    maxiter : int
        Maximum allowed number of iterations to reach convergence.

    Returns
    -------
    active_events : ndarray, shape (n, )
        Array containing the indexes of each active event. It may
        differ from the input `active_events` array because only
        the active events that happen before the first triggered terminal
        event are returned.
    roots : ndarray, shape (n, )
        Event times of all the active events returned.
    terminate : boolean
        True if one of the active events is terminal.
    converged: boolean 
        True if all the active events reached convergence.
    """

    roots = np.zeros((len(active_events,)))
    for k, idx in enumerate(active_events):
        converged, roots[k] = ridder(events[idx], yk, args, Q, DOP,
                             tk_old, tk, event_tol, maxiter)

        if not converged: 
            return active_events, roots, True, False

    # If there are any terminal events, only the ones that happen before the
    # first terminal event are returned
    if np.any(is_terminal[active_events]):
        if tk > tk_old:
            order = np.argsort(roots)
        else:
            order = np.argsort(-roots)

        # Active events are ordered:
        active_events = active_events[order]
        roots = roots[order]

        # Retrieves the timing of the first active terminal event
        t = np.nonzero(is_terminal[active_events])[0][0]
        active_events = active_events[:t + 1]
        roots = roots[:t + 1]
        terminate = True
    else:
        terminate = False

    return active_events, roots, terminate, True

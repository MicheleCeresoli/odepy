import warnings
from typing import Callable, List, Tuple

import numpy as np
from numba import jit, types
from numba.core.errors import NumbaExperimentalFeatureWarning
from numba.typed import List as nbList

from odepy.controllers import H312PID, controller_signature as cnts
from odepy.dense import dense_output, interpolate
from odepy.events import assemble_events, get_active_events, handle_events
from odepy.events import event_wrapper_signature as ews
from odepy.step import compute_initial_step, perform_step
from odepy.tableaus import Vern9, rk_method_signature as rkms
from odepy.utils import isclose, dfun_signature, vfloat_sig, jit_settings

# Disables experimental warning issue
warnings.simplefilter('ignore', category=NumbaExperimentalFeatureWarning)

float_list_sig = types.ListType(types.float64)
vfloat_list_sig = types.ListType(vfloat_sig)

_int_outputs = ("Tuple((float64[::1], float64[:, ::1], ListType(ListType(float64)), " +
                "ListType(ListType(float64[::1])), int64))")

_int_inputs = ("(" + dfun_signature + ", float64[::1], float64[::1]," +
                   "float64[::1], FunctionType(" + rkms + ")," +
                   "FunctionType(" + cnts + "), float64, float64)")

_int_const_sign = _int_outputs + _int_inputs[:-1] + ", "

@jit([_int_const_sign + "float64, " + ews + ")",
      _int_const_sign + "Omitted(None), Omitted(None))"],
      **jit_settings)
def solver(fun: Callable[[np.ndarray, float, np.ndarray, np.ndarray], None],
           tspan: np.ndarray, y0: np.ndarray, args: np.ndarray, method: Callable,
           controller: Callable, atol: float, rtol: float, events_tol: float = None,
           events: List[Callable] = None) -> Tuple[np.ndarray, np.ndarray,
                                                   List[List[float]],
                                                   List[List[np.ndarray]], int]:

    """Integrate an Ordinary Differential Equation (ODE).

    This function integrates a system of ordinary differential
    equations in the form of::

        dt / dt = f(t, y, args)
        y(t0) = y0

    Here, `t` is the independent (time) variable, `y(t)` is
    the state vector and `args` is a vector of constant parameters
    that the equations may depend on and `y0` is the initial
    state value.

    This function finds a solution to the above problem using a
    user-specified Runge-Kutta explicit method. An adaptive
    step-size control is used to adjust the integration step
    after each RK iteration step.

    Parameters
    ----------
    fun : callable
        Function representing the system of ODEs to integrate.
        The function must be jitted and have the following calling
        signature: ``fun(dy, t, y, args)``, where `dy`, `y`, and
        `args` are C-contiguous float64-Numpy arrays. The in-placing
        of `dy` means that the function shall not return anything.
        This choice avoids creating a new array every time the
        function is called, minimising the overall integration time.
    tspan : ndarray, shape (npnts, )
        Desired solution points. The initial and final values
        are considered the extrems of the integration interval.
        If only (t0, tf) are specified, the solver will only return
        the solution at the final integration time (tf). Otherwise,
        it will return the dense-output solution at the specified
        times. This array must either be sorted in ascending
        (forward-integration) or descending (backwards-integration)
        order.
    y0 : ndarray, shape (ndof, )
        Initial state.
    args : ndarray, shape (nargs, )
        Additional ODE and event function parameters. If those
        functions do not depend on any parameter, an empty float-64
        array shall be provided nonetheless.
    method : Callable
        Desired explicit Runge-Kutta solver. The function returns
        the tableau associated to the user-desired RK method.
    controller : Callable
        Desired adaptive step-size controller.
    atol, rtol : float
        Absolute and relative integration tolerances. The solver
        keeps the local error estimates smaller than
        ``atol + rtol * abs(y)``. Note that both values are
        local tolerances and thus are not global guarantees that
        the error with respect to the true solution will always be
        smaller than the desired value. If extremely small values are
        provided (smaller than machine precision), the solver may
        fail to perform an integration step.
    events_tol : float, optional
        Absolute tolerance for the event root-finding algorithm.
    events : Typed-list, optional
        A typed-list of jitted event functions. Each function shall
        have the signature: ``event(t, y, args)`` and return a
        tuple of three arguments:

            value : float
                The result of an expression which describes the event.
                The event happens when value is equal to zero.
            isterminal : int
                Specifies whether the integration shall be stopped if
                this specific event occurs. A value of 0 will
                terminate the integration on occurance.
            direction : int
                Direction of a zero crossing. If direction = 0 all
                zeros will be located. A positive `direction` will
                trigger the event only when the event `value` goes
                from negative to positive and viceversa if
                `direction` is negative.

        The solver does not support changes to the returned values
        of `isterminal` and `direction` during integration.
        The main routine determines the terminaly and direction
        properties of each event only at the beginning of the
        integration. Therefore, later changes of their
        values will be neglected.

    Returns
    -------
    th : ndarray, shape (n, )
        Time points at which the solution has been computed.
    yh : ndarray, shape (n, ndof)
        Values of the solution at `t`.
    t_events : list of list of float
        It contains for each event a list of values at which
        the event was detected. If no events were detected,
        an empty list is returned.
    y_events : list of list of ndarray
        It contains the solution associated to each value of
        `t_events`.
    e_flag : int
        Exit flag:
         * 0: The solver successfully reached the end of `tspan`.
         * 1: The integration failed because the required step-size
              is below the minimum allowed size.
         * 2: The integration could not begin. The desired relative
              tolerance is below the machine precision.
         * 3: A terminal event has been activated. The integration
              successfully stopped before reaching the end of `tspan`.
         * 4: The integration stopped because the solver was unable to 
              reach convergence on the root of an active event.
            
    Notes
    -----
    The function does not currently support integration in the
    complex domain.
    """

    # Event max iteration for convergence
    events_maxiter = 50

    e_flag = 0

    hmin = 16*np.finfo(np.float64).eps
    hmax = abs(tspan[-1] - tspan[0])
    dense = tspan.shape[0] > 2

    # RK Method parameters
    params = method()
    isDOP = params[4][0] != 0
    # A, B, C, E, E3, P = params[:6]
    # FSAL, order, error_estimation_order, stages = params[6:10]
    # extra_stages = params[-1]

    # Controller parameters
    betas = controller()
    betas *= -1/(params[8] + 1)

    ndof = len(y0)
    K = np.empty((params[9], ndof))

    fk = np.empty_like(y0)
    fun(fk, tspan[0], y0, args)

    t_dir = np.sign(tspan[-1]-tspan[0])

    absh = compute_initial_step(
        fun, tspan, y0, args, fk, ndof, atol, rtol, params[8])

    if dense:
        th = tspan
        yh = np.empty((th.shape[0], ndof), dtype=np.float64) # Pre-allocates array
        yh[0] = y0
        dense_cnt = 1

    tk = tspan[0]
    yk = y0

    # Outside of event management to initialise outputs
    t_events = nbList.empty_list(float_list_sig)
    y_events = nbList.empty_list(vfloat_list_sig)

    # Event management
    if events is not None:
        n_events = len(events)

        evt_val = np.empty((n_events,))
        evt_new_val = np.empty_like(evt_val)

        evt_terminal = np.empty((n_events,), dtype=np.int64)
        evt_direction = np.empty_like(evt_terminal)

        for i in range(n_events):
            evt_val[i], evt_terminal[i], evt_direction[i] = events[i](tk, yk, args)

        # Sets the sub-list for each event function
        for _ in range(n_events):
            t_events.append(nbList.empty_list(types.float64))
            y_events.append(nbList.empty_list(vfloat_sig))

    # Controls the relative tolerances is not below the machine precision
    if rtol < hmin:
        e_flag = 2
        return np.asarray([tk]), np.atleast_2d(yk), t_events, y_events, e_flag

    errs = -np.ones((3,))

    ode_complete = False
    while not ode_complete:

        absh = min(hmax, max(hmin, absh))
        tkp, ykp, absh, e_flag = perform_step(
            fun, tk, yk, args, fk, absh, t_dir, tspan[-1],
            atol, rtol, K, hmin, errs, betas, params
        )

        if e_flag != 0:
            break

        h_old = tkp - tk

        if isclose(tkp, tspan[-1], atol, rtol):
            ode_complete = True
            tkp = tspan[-1] # to avoid issues with the dense-output comparison

        if dense: # Computes dense-output coeffficients
            Q = dense_output(
                fun, tk, yk, args, ykp, fk, h_old, K, params[0],
                params[2], params[5], params[9], params[-1], isDOP)

        if events is not None:
            for i in range(n_events):
                evt_new_val[i] = events[i](tkp, ykp, args)[0]

            active_events = get_active_events(evt_val, evt_new_val, evt_direction)
            if len(active_events) > 0:
                if not dense:
                    Q = dense_output(
                        fun, tk, yk, args, ykp, fk, h_old, K, params[0],
                        params[2], params[5], params[9], params[-1], isDOP)

                root_idx, roots, terminate, converged = handle_events(
                    events, active_events, evt_terminal, yk, args,
                    Q, isDOP, tk, tkp, events_tol, events_maxiter)

                # The root of at least one active event could not be found.
                if not converged:
                    e_flag = 4 
                    break

                for e, te in zip(root_idx, roots):
                    t_events[e].append(te)
                    y_events[e].append(interpolate(te, yk, tk, h_old, Q, isDOP)[0])

                if terminate:
                    ode_complete = True
                    e_flag = 3
                    tkp = roots[-1]
                    ykp = y_events[root_idx[-1]][-1]

            evt_val = np.copy(evt_new_val)

        if dense: # -> vedi se c'è qualche valore in tspan compreso tra tk e tkp
            if t_dir > 0:
                tspan = tspan[tspan > tk]
                tms = tspan[tspan <= tkp]

            else:
                tspan = tspan[tspan < tk]
                tms = tspan[tspan >= tkp]

            if tms.shape[0] > 0:
                yh[dense_cnt:dense_cnt+tms.shape[0]] = interpolate(tms, yk, tk,
                                                                   h_old, Q, isDOP)
                dense_cnt += tms.shape[0]

            if ode_complete:
                if t_dir > 0:
                    th = th[th <= tkp]
                else:
                    th = th[th >= tkp]

                yh = yh[:th.shape[0]]

        tk = tkp
        yk = ykp

    if not dense:
        th = np.asarray([tk])
        yh = np.atleast_2d(yk)

    return th, yh, t_events, y_events, e_flag


@jit("Tuple((float64[::1], float64[:, ::1], int64))" + _int_inputs, **jit_settings)
def fast_ivp(fun: Callable[[np.ndarray, float, np.ndarray, np.ndarray], None],
           tspan: np.ndarray, y0: np.ndarray, args: np.ndarray, method: Callable,
           controller: Callable, atol: float, rtol: float) -> Tuple[float, np.ndarray, int]:
    """Integrates a system of Ordinary Differential Equations (ODE) without events.

    This function integrates a system of ordinary differential
    equations in the form of::

        dt / dt = f(t, y, args)
        y(t0) = y0

    Here, `t` is the independent (time) variable, `y(t)` is
    the state vector and `args` is a vector of constant parameters
    that the equations may depend on and `y0` is the initial
    state value.

    Parameters
    ----------
    fun : callable
        Jitted function representing the system of ODEs to integrate.
        The function must have the following calling signature:
        ``fun(dy, t, y, args)``, where `dy`, `y`, and `args` are
        C-contiguous float64-Numpy arrays.
    tspan : ndarray, shape (npnts, )
        Desired solution points. The initial and final values
        are considered the extrems of the integration interval.
        If only (t0, tf) are specified, the solver will only return
        the solution at the final integration time (tf). Otherwise,
        it will return the dense-output solution at the specified
        times. This array must either be sorted in ascending
        (forward-integration) or descending (backwards-integration)
        order.
    y0 : ndarray, shape (ndof, )
        Initial state.
    args : ndarray, shape (nargs, )
        Additional ODE and event function parameters. If those
        functions do not depend on any parameter, an empty float-64
        array shall be provided nonetheless.
    method : Callable
        Desired explicit Runge-Kutta solver. The function returns
        the tableau associated to the user-desired RK method.
    controller : Callable
        Desired adaptive step-size controller.
    atol, rtol : float
        Absolute and relative integration tolerances. The solver
        keeps the local error estimates smaller than
        ``atol + rtol * abs(y)``. Note that both values are
        local tolerances and thus are not global guarantees that
        the error with respect to the true solution will always be
        smaller than the desired value. If extremely small values are
        provided (smaller than machine precision), the solver may
        fail to perform an integration step.

    Returns
    -------
    th : ndarray, shape (n, )
        Time points at which the solution has been computed.
    yh : ndarray, shape (n, ndof)
        Values of the solution at `t`. If `tspan` contains
        only 2 values, only the final integration state
        is returned.
    e_flag : int
        Exit flag:
         * 0: The solver successfully reached the end of `tspan`.
         * 1: The integration failed because the required step-size
              is below the minimum allowed size.
         * 2: The integration could not begin. The desired relative
              tolerance is below the machine precision.
         * 3: A terminal event has been activated. The integration
              successfully stopped before reaching the end of `tspan`.
         * 4: The integration stopped because the solver was unable to 
              reach convergence on the root of an active event.
    Notes
    -----
    The function does not currently support integration in the
    complex domain.
    """

    t, y, _, _, e_flag = solver(fun, tspan, y0, args, method, controller, atol, rtol)
    return t, y, e_flag

@jit(_int_const_sign + ews + ", float64)", **jit_settings)
def fast_ivpe(fun: Callable[[np.ndarray, float, np.ndarray, np.ndarray], None],
           tspan: np.ndarray, y0: np.ndarray, args: np.ndarray, method: Callable,
           controller: Callable, atol: float, rtol: float, events: List[Callable],
           events_tol: float) -> Tuple[np.ndarray, np.ndarray,
                                              List[List[float]],
                                              List[List[np.ndarray]], int]:
    """Integrates a system of Ordinary Differential Equations (ODE) with events.

    This function integrates a system of ordinary differential
    equations in the form of::

        dt / dt = f(t, y, args)
        y(t0) = y0

    Here, `t` is the independent (time) variable, `y(t)` is
    the state vector and `args` is a vector of constant parameters
    that the equations may depend on and `y0` is the initial
    state value.

    Parameters
    ----------
    fun : callable
        Jitted function representing the system of ODEs to integrate.
        The function must have the following calling signature:
        ``fun(dy, t, y, args)``, where `dy`, `y`, and `args` are
        C-contiguous float64-Numpy arrays.
    tspan : ndarray, shape (npnts, )
        Desired solution points. The initial and final values
        are considered the extrems of the integration interval.
        If only (t0, tf) are specified, the solver will only return
        the solution at the final integration time (tf). Otherwise,
        it will return the dense-output solution at the specified
        times. This array must either be sorted in ascending
        (forward-integration) or descending (backwards-integration)
        order.
    y0 : ndarray, shape (ndof, )
        Initial state.
    args : ndarray, shape (nargs, )
        Additional ODE and event function parameters. If those
        functions do not depend on any parameter, an empty float-64
        array shall be provided nonetheless.
    method : Callable
        Desired explicit Runge-Kutta solver. The function returns
        the tableau associated to the user-desired RK method.
    controller : Callable
        Desired adaptive step-size controller.
    atol, rtol : float
        Absolute and relative integration tolerances. The solver
        keeps the local error estimates smaller than
        ``atol + rtol * abs(y)``. Note that both values are
        local tolerances and thus are not global guarantees that
        the error with respect to the true solution will always be
        smaller than the desired value. If extremely small values are
        provided (smaller than machine precision), the solver may
        fail to perform an integration step.
    events : Typed-list
        A typed-list of jitted event functions. Such list is created
        by invoking the `assemble_events` function, which accepts
        as inputs a tuple of event functions. Each of these functions
        shall have the signature: ``event(t, y, args)`` and return
        three arguments:

            value : float
                The result of an expression which describes the event.
                The event happens when value is equal to zero.
            isterminal : int
                Specifies whether the integration shall be stopped if
                this specific event occurs. A value of 0 will
                terminate the integration on occurance.
            direction : int
                Direction of a zero crossing. If direction = 0 all
                zeros will be located. A positive `direction` will
                trigger the event only when the event `value` goes
                from negative to positive and viceversa if
                `direction` is negative.

        The solver does not support changes to the returned values
        of `isterminal` and `direction` during integration.
        The main routine determines the terminaly and direction
        properties of each event only at the beginning of the
        integration. Therefore, later changes of their
        values will be neglected.
    events_tol : float
        Absolute tolerance for the event root-finding algorithm.

    Returns
    -------
    th : ndarray, shape (n, )
        Time points at which the solution has been computed.
    yh : ndarray, shape (n, ndof)
        Values of the solution at `t`. If `tspan` contains
        only 2 values, only the final integration state
        is returned.
    t_events : list of list of float
        It contains for each event a list of values at which
        the event was detected. If no events were detected,
        an empty list is returned.
    y_events : list of list of ndarray
        It contains the solution associated to each value of
        `t_events`.
    e_flag : int
        Exit flag:
         * 0: The solver successfully reached the end of `tspan`.
         * 1: The integration failed because the required step-size
              is below the minimum allowed size.
         * 2: The integration could not begin. The desired relative
              tolerance is below the machine precision.
         * 3: A terminal event has been activated. The integration
              successfully stopped before reaching the end of `tspan`.
         * 4: The integration stopped because the solver was unable to 
              reach convergence on the root of an active event.
    Notes
    -----
    The function does not currently support integration in the
    complex domain.

    """

    return solver(fun, tspan, y0, args, method, controller, atol, rtol,
                  events_tol, events)



def solve(fun: Callable[[np.ndarray, float, np.ndarray, np.ndarray], None],
           tspan: np.ndarray, y0: np.ndarray, args: np.ndarray = None,
           method: Callable = Vern9, controller: Callable = H312PID,
           atol: float = 1e-5, rtol: float = 1e-4, events: List[Callable] = None,
           events_tol: float = 1e-10) -> Tuple[np.ndarray, np.ndarray,
                                               List[List[float]],
                                               List[List[np.ndarray]], int]:

    """Integrates a system of Ordinary Differential Equations (ODE).

    This function integrates a system of ordinary differential
    equations in the form of::

        dt / dt = f(t, y, args)
        y(t0) = y0

    Here, `t` is the independent (time) variable, `y(t)` is
    the state vector and `args` is a vector of constant parameters
    that the equations may depend on and `y0` is the initial
    state value.

    Parameters
    ----------
    fun : callable
        Jitted function representing the system of ODEs to integrate.
        The function must have the following calling signature:
        ``fun(dy, t, y, args)``, where `dy`, `y`, and `args` are
        C-contiguous float64-Numpy arrays.
    tspan : ndarray, shape (npnts, )
        Desired solution points. The initial and final values
        are considered the extrems of the integration interval.
        If only (t0, tf) are specified, the solver will only return
        the solution at the final integration time (tf). Otherwise,
        it will return the dense-output solution at the specified
        times. This array must either be sorted in ascending
        (forward-integration) or descending (backwards-integration)
        order.
    y0 : ndarray, shape (ndof, )
        Initial state.
    args : ndarray, shape (nargs, ), optional
        Additional ODE and event function parameters. If those
        functions do not depend on any parameter, an empty float-64
        array shall be provided nonetheless.
    method : Callable, optional
        Desired explicit Runge-Kutta solver. The function returns
        the tableau associated to the user-desired RK method. Default
        is Vern9.
    controller : Callable, optional
        Desired adaptive step-size controller. Default is H312PID.
    atol, rtol : float, optional
        Absolute and relative integration tolerances. The solver
        keeps the local error estimates smaller than
        ``atol + rtol * abs(y)``. Note that both values are
        local tolerances and thus are not global guarantees that
        the error with respect to the true solution will always be
        smaller than the desired value. If extremely small values are
        provided (smaller than machine precision), the solver may
        fail to perform an integration step.
    events : list, optional
        A list of jitted event functions. Such list is created
        by invoking the `assemble_events` function, which accepts
        as inputs a tuple of event functions. Each of these functions
        shall have the signature: ``event(t, y, args)`` and return
        three arguments:

            value : float
                The result of an expression which describes the event.
                The event happens when value is equal to zero.
            isterminal : int
                Specifies whether the integration shall be stopped if
                this specific event occurs. A value of 0 will
                terminate the integration on occurance.
            direction : int
                Direction of a zero crossing. If direction = 0 all
                zeros will be located. A positive `direction` will
                trigger the event only when the event `value` goes
                from negative to positive and viceversa if
                `direction` is negative.

        The solver does not support changes to the returned values
        of `isterminal` and `direction` during integration.
        The main routine determines the terminaly and direction
        properties of each event only at the beginning of the
        integration. Therefore, later changes of their
        values will be neglected.
    events_tol : float, optional
        Absolute tolerance for the event root-finding algorithm.

    Returns
    -------
    th : ndarray, shape (n, )
        Time points at which the solution has been computed.
    yh : ndarray, shape (n, ndof)
        Values of the solution at `t`. If `tspan` contains
        only 2 values, only the final integration state
        is returned.
    t_events : list of list of float
        It contains for each event a list of values at which
        the event was detected. If no events were detected,
        an empty list is returned.
    y_events : list of list of ndarray
        It contains the solution associated to each value of
        `t_events`.
    e_flag : int
        Exit flag:
         * 0: The solver successfully reached the end of `tspan`.
         * 1: The integration failed because the required step-size
              is below the minimum allowed size.
         * 2: The integration could not begin. The desired relative
              tolerance is below the machine precision.
         * 3: A terminal event has been activated. The integration
              successfully stopped before reaching the end of `tspan`.
         * 4: The integration stopped because the solver was unable to 
              reach convergence on the root of an active event.
    Notes
    -----
    The function does not currently support integration in the
    complex domain.

    """

    tspan = np.asarray(tspan)
    y0 = np.asarray(y0)

    if args is None:
        args = np.array([0.])

    if events is None:
        sol = solver(fun, tspan, y0, args, method, controller, atol, rtol)
        return sol[0], sol[1], sol[-1]
    else:
        return solver(fun, tspan, y0, args, method, controller, atol, rtol,
                      events_tol, assemble_events(tuple(events)))

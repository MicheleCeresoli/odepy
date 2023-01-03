import numpy as np
from numba import jit

import odepy as op
import odepy.tableaus as tb
import odepy.controllers as ct

jit_settings = {'nopython': True, 'nogil': True}

# A1
@jit(op.fun_signature, **jit_settings)
def A1(dy, t, y, args):
    dy[0] = -y[0]

def A1e(tms, y0):
    y = y0*np.exp(-(tms-tms[0]))
    return y

@jit(op.fun_signature, **jit_settings)
def A2(dy, t, y, args):
    dy[0] = -0.5*y[0]**3

def A2e(tms, y0):
    A = 1/y0**2 - tms[0]
    y = 1/np.sqrt(tms + A)
    return y

# A3
@jit(op.fun_signature, **jit_settings)
def A3(dy, t, y, args):
    dy[0] = np.cos(t)*y[0]

def A3e(tms, y0):
    y = y0*np.exp(np.sin(tms) - np.sin(tms[0]))
    return y

problems =  [(A1, A1e), (A2, A2e), (A3, A3e)]

tableaus = [tb.BS5, tb.Vern7, tb.Vern8,
            tb.Vern9, tb.DOP853]

controllers = [ct.H211PI, ct.H312PID, ct.integral_controller,
               ct.PI33, ct.PI34, ct.PI42]

def test_1d_forward():

    y0 = np.array([2.])
    args = np.array([0.])
    tspan = np.array([1., 3.])

    atol, rtol = 1e-10, 1e-10

    for tab in tableaus:
        for cnt in controllers:
            for fcn, sol in problems:
                t, y, _ = op.fast_ivp(fcn, tspan, y0, args, tab, cnt, atol, rtol)

                assert t == tspan[-1]
                assert np.isclose(y, sol(tspan, y0)[-1])

def test_1d_backward():

    y0 = np.array([0.1])
    args = np.array([0.])
    tspan = np.array([6., 4.])

    atol, rtol = 1e-10, 1e-10

    for tab in tableaus:
        for cnt in controllers:
            for fcn, sol in problems:
                t, y, _ = op.fast_ivp(fcn, tspan, y0, args, tab, cnt, atol, rtol)

                assert t == tspan[-1]
                assert np.isclose(y, sol(tspan, y0)[-1])


@jit(op.fun_signature, **jit_settings)
def fcn(dy, t, y, args):
    for i in range(len(y)):
        dy[i] = -y[i]

def test_2d_dense_forward():

    y0 = np.array([3., 4., 2., 7.])
    tspan = np.linspace(-2, 10., 20)
    args = np.array([0.])

    atol, rtol = 1e-10, 1e-10

    for tab in tableaus:
        for cnt in controllers:
            t, y, _ = op.fast_ivp(fcn, tspan, y0, args, tab, cnt, atol, rtol)

            assert np.all(t == tspan)
            assert y.shape[0] == len(tspan)
            assert y.shape[1] == len(y0)

            for i in range(len(y0)):
                err = np.abs(y[:, i] - y0[i]*np.exp(-(tspan-tspan[0])))
                assert np.max(err) < 1e-2


def test_2d_dense_backward():

    y0 = np.array([0.4, 0.42, .485, .002])
    tspan = np.linspace(-0, -2., 20)
    args = np.array([0.])

    atol, rtol = 1e-10, 1e-10

    for tab in tableaus:
        for cnt in controllers:
            t, y, _ = op.fast_ivp(fcn, tspan, y0, args, tab, cnt, atol, rtol)

            assert np.all(t == tspan)
            assert y.shape[0] == len(tspan)
            assert y.shape[1] == len(y0)

            for i in range(len(y0)):
                err = np.abs(y[:, i] - y0[i]*np.exp(-(tspan-tspan[0])))
                assert np.max(err) < 1e-2

def test_exit_flag():

    y0 = np.array([0.4, 0.42, .485, .002])
    tspan = np.linspace(-0, -2., 20)
    args = np.array([0.])

    atol, rtol = 1e-10, 1e-22

    t, y, e_flag = op.fast_ivp(fcn, tspan, y0, args, tb.Vern9, ct.H312PID, atol, rtol)
    assert e_flag == 2

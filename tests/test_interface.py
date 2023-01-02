import numpy as np
from numba import jit

import odepy as rp

jit_settings = {'nopython': True, 'nogil': True}

@jit(rp.fun_signature, **jit_settings)
def expo(dy, t, y, args):
    dy[0] = -y[0]

@jit(rp.event_signature, **jit_settings)
def e1(t, y, args):
    return 1., 1., 0.

@jit(rp.event_signature, **jit_settings)
def e2(t, y, args):
    return t-10., 1., 0.

@jit(rp.event_signature, **jit_settings)
def e3(t, y, args):
    return t+2.5, 1., 0.

def test_scalar_outputs_shape():

    args = np.array([0.])
    tspan = np.array([0., 1.])
    y0 = np.array([2.])

    atol, rtol = 1e-10, 1e-10
    etol = 1e-10

    t, y, _ = rp.fast_ivp(expo, tspan, y0, args, rp.Vern9, rp.H312PID, atol, rtol)
    te, ye, *_ = rp.fast_ivpe(expo, tspan, y0, args, rp.Vern9, rp.H312PID, atol, rtol,
                                rp.assemble_events((e1,)), etol)
    assert t.shape[0] == 1
    assert y.shape[0] == 1
    assert y.shape[1] == 1

    assert te.shape[0] == 1
    assert ye.shape[0] == 1
    assert ye.shape[1] == 1

def test_dense_outputs_shape():

    args = np.array([0.])
    tspan = np.linspace(0., 100., 100)
    y0 = np.array([2.])

    atol, rtol = 1e-10, 1e-10
    etol = 1e-10

    t, y, _ = rp.fast_ivp(expo, tspan, y0, args, rp.Vern9, rp.H312PID, atol, rtol)
    te, ye, *_ = rp.fast_ivpe(expo, tspan, y0, args, rp.Vern9, rp.H312PID, atol, rtol,
                                rp.assemble_events((e1,)), etol)
    assert t.shape[0] == 100
    assert y.shape[0] == 100
    assert y.shape[1] == 1

    assert te.shape[0] == 100
    assert ye.shape[0] == 100
    assert ye.shape[1] == 1

def test_dense_events_shape():

    args = np.array([0.])
    tspan = np.linspace(0., 100., 100)
    y0 = np.array([2.])

    atol, rtol = 1e-10, 1e-10
    etol = 1e-10

    te, ye, *_ = rp.fast_ivpe(expo, tspan, y0, args, rp.Vern9, rp.H312PID, atol, rtol,
                        rp.assemble_events((e2,)), etol)

    npnts = len(te[te <= 10.])
    assert te.shape[0] == npnts
    assert ye.shape[0] == npnts
    assert ye.shape[1] == 1

    tspan = np.linspace(0., -5., 100)
    te, ye, *_ = rp.fast_ivpe(expo, tspan, y0, args, rp.Vern9, rp.H312PID, atol, rtol,
                            rp.assemble_events((e3,)), etol)

    npnts = len(te[te >= -2.5])
    assert te.shape[0] == npnts
    assert ye.shape[0] == npnts
    assert ye.shape[1] == 1

def test_python_shape():

    tspan = np.array([0., 1.])
    y0 = np.array([2.])

    atol, rtol = 1e-10, 1e-10

    t, y, _ = rp.solve(expo, tspan, y0, atol=atol, rtol=rtol)
    te, ye, *_ = rp.solve(expo, tspan, y0, events=[e1])

    assert t.shape[0] == 1
    assert y.shape[0] == 1
    assert y.shape[1] == 1

    assert te.shape[0] == 1
    assert ye.shape[0] == 1
    assert ye.shape[1] == 1

import numpy as np
from numba import jit

import odepy as rp

jit_settings = {'nopython': True, 'nogil': True}

@jit(rp.fun_signature, **jit_settings)
def expo(dy, t, y, args):
    dy[0] = -y[0]

@jit(rp.event_signature, **jit_settings)
def e1(t, y, args):
    return y[0], 1., 0.

@jit(rp.event_signature, **jit_settings)
def e2(t, y, args):
    return t, 1., 0.

@jit(rp.event_signature, **jit_settings)
def e21(t, y, args):
    return t, 1., 1.

@jit(rp.event_signature, **jit_settings)
def e22(t, y, args):
    return t, 1., -1.

@jit(rp.event_signature, **jit_settings)
def e23(t, y, args):
    return t-0.1, 0., 1.

@jit(rp.event_signature, **jit_settings)
def e24(t, y, args):
    return t-2., 1., 1.


def test_assemble():

    events = rp.assemble_events((e1,))
    assert len(events) == 1

    events = rp.assemble_events((e1, e2))
    assert len(events) == 2

    assert events[-1](-99., np.array([2.]), np.array([0.]))[0] == -99

def test_events():

    args = np.array([0.])
    tspan = np.array([-2, 10.])
    y0 = np.array([2.])

    atol, rtol = 1e-10, 1e-10
    etol = 1e-10

    # Test event solution
    t, y, te, ye, _ = rp.fast_ivpe(expo, tspan, y0, args, rp.Vern9, rp.H312PID, atol, rtol,
                                rp.assemble_events((e2,)), etol)

    assert t == 0.
    assert np.isclose(y, y0*np.exp(-2)) == True
    assert t == te[0][0]
    assert y[0] == ye[0][0]

    tspan = np.array([0.1, 10.])
    t, y, *_ = rp.fast_ivpe(expo, tspan, y0, args, rp.Vern9, rp.H312PID, atol, rtol,
                                rp.assemble_events((e21,)), etol)
    assert t == 10.

    # Test directions
    t, y, *_ = rp.fast_ivpe(expo, tspan, y0, args, rp.Vern9, rp.H312PID, atol, rtol,
                                rp.assemble_events((e22,)), etol)
    assert t == 10.

    # Test non terminal events
    t, y, te, ye, _ = rp.fast_ivpe(expo, tspan, y0, args, rp.Vern9, rp.H312PID, atol, rtol,
                                rp.assemble_events((e21,e23)), etol)
    assert t == 10.
    assert te[-1][0] == 0.1

    # Test terminality flags
    rtol = 1e-10
    t, y, _, _, e_flag = rp.fast_ivpe(expo, tspan, y0, args, rp.Vern9, rp.H312PID, atol, rtol,
                                rp.assemble_events((e24,)), etol)
    assert e_flag == 3
    assert np.isclose(t, 2)

    rtol = 1e-22
    t, y, _, _, e_flag = rp.fast_ivpe(expo, tspan, y0, args, rp.Vern9, rp.H312PID, atol, rtol,
                                rp.assemble_events((e21,e23)), etol)
    assert e_flag == 2



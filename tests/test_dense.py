import numpy as np

from odepy.dense import interpolate


def test_scalar_interpolate():

    yk = np.array([0.])
    tk, hk = 0., 1.

    DOP = False
    Q = np.array([[1.], [1.]])

    ts = 10.
    y = interpolate(ts, yk, tk, hk, Q, DOP)

    sol = tk + Q[0,0]*ts + Q[1,0]*ts**2

    assert y.ndim == 2
    assert y.shape[0] == 1
    assert y.shape[1] == 1
    assert y[0, 0] == sol


def test_backward_interpolate():
    yk = np.array([0.])
    tk, hk = 0., 1.

    DOP = False

    Q = np.array([[3], [4.], [-5]])

    sol = lambda x: Q[0]*x + Q[1]*x**2 + Q[2]*x**3

    tms = np.linspace(0., -10., 25)
    y = interpolate(tms, yk, tk, hk, Q, DOP)

    assert y.shape[0] == 25
    assert y.shape[1] == 1

    error = np.abs(y.T[0] - sol(tms))
    assert np.max(error) < 1e-9

def test_forward_interpolate():

    yk = np.array([0.])
    tk, hk = 0., 1.

    DOP = False

    Q = np.array([[3], [4.], [-5]])

    sol = lambda x: Q[0]*x + Q[1]*x**2 + Q[2]*x**3

    tms = np.linspace(0., 10., 25)
    y = interpolate(tms, yk, tk, hk, Q, DOP)

    assert y.shape[0] == 25
    assert y.shape[1] == 1

    error = y.T[0] - sol(tms)
    assert np.max(error) < 1e-9


def test_2d_interpolate():

    yk = np.array([0., 4.])
    tk, hk, = 0., 1.

    DOP = False

    Q = np.array([[3., 7.], [4., 2.], [-5., -2.]])

    sol = lambda x, i: yk[i] + Q[0,i]*x + Q[1,i]*x**2 + Q[2,i]*x**3

    tms = np.linspace(0., 10., 100)
    y = interpolate(tms, yk, tk, hk, Q, DOP)

    assert y.shape[0] == len(tms)
    assert y.shape[1] == len(yk)

    e1 = np.abs(y.T[0] - sol(tms, 0))
    e2 = np.abs(y.T[1] - sol(tms, 1))

    assert np.max(e1) < 1e-10
    assert np.max(e2) < 1e-10

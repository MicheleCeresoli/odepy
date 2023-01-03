import numpy as np

from odepy import controllers as ctl

from odepy.controllers import SAFETY, MIN_FACTOR, MAX_FACTOR


cnt = [ctl.H211PI, ctl.H312PID, ctl.PI33,
       ctl.PI34, ctl.PI42, ctl.integral_controller]

def test_controllers_shapes():
    for c in cnt:
        assert c().shape[0] == 3

def test_integral_control():
    # Integral control
    errs = np.array([3., -1., 5.])
    betas = np.array([0.002, 4., 1.])

    factor, result =  ctl._basic_control(errs, betas, False)
    assert result == True

    assert factor == max(MIN_FACTOR, SAFETY*errs[-1]**betas[0])

    factor, result =  ctl._basic_control(np.array([3., -1., 0]),
                                         betas, False)
    assert result == False
    assert factor == 0.


def test_PI_control():

    # PI Control
    errs = np.array([4., 1e-5, 1e-8])
    betas = np.array([0., 0.04, -0.004])

    factor, result =  ctl._basic_control(errs, betas, False)
    assert result == False
    assert factor == min(MAX_FACTOR,
                         SAFETY*errs[-1]**betas[0]*errs[-2]**betas[1])


def test_PID_control():

    errs = np.array([4., 1e-5, 1e-8])
    betas = np.array([0., 0.04, -0.004])

    factor = errs[0]**betas[-1]*errs[1]**betas[-2]*errs[2]**betas[0]
    factor = 1 + np.arctan(factor - 1)

    result, _ = ctl._pid_control(errs, betas)

    assert result == factor

    _, outcome = ctl._pid_control(100*np.ones((3,)), betas)
    assert outcome == False


def test_stepsize_controller():
    errs = np.zeros((3,))

    factor, _ = ctl.stepsize_controller(errs, np.ones((3,)), False)
    assert factor == MAX_FACTOR

    betas = np.array([3., 0., 0.])
    errs = np.array([1e-3, 1e-5, 1e-2])
    factor, _ = ctl.stepsize_controller(errs, betas, False)
    result, _ = ctl._basic_control(errs, betas, False)
    assert factor == result

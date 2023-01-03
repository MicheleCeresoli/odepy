import numpy as np

import odepy.tableaus as tb

tableaus = [tb.BS5, tb.Vern7, tb.Vern8,
            tb.Vern9, tb.DOP853]

def test_tableaus():

    for tab in tableaus:
        x = tab()
        assert len(x) == 11

    A, B, C, E, E3, P = x[:6]
    FSAL, order, error_estimation_order, stages = x[6:10]
    extra_stages = x[10:]

    stages, extra_stages = x[9], x[-1]
    FSAL = x[6]

    DOP = A[6, 0] == 3.7109375e-2

    assert A.ndim == 2
    assert (A.shape[0] == stages + extra_stages
            and A.shape[1] == stages+extra_stages)

    assert B.ndim == 1
    if FSAL:
        assert x[1].shape[0] == 1
    else:
        assert x[1].shape[0] == stages

    assert C.ndim == 1 and (C.shape[0] == stages + extra_stages)
    assert E.ndim == 1 and E.shape[0] == stages

    assert E3.ndim == 1

    if not DOP:
        assert E3[0] == 0.

    assert P.ndim == 2
    if not DOP:
        assert P.shape[1] == stages + extra_stages



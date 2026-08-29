# Baseline iterative tests for EvoEval_creative/22

def baseline_check_0(candidate):
    import math
    res = candidate(1, 300000)
    assert isinstance(res, (int, float))
    assert math.isclose(res, 0.9999788605855618, rel_tol=0.0, abs_tol=1e-12)

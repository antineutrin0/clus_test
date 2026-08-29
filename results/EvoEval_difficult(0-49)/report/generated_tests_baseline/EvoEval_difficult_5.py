# Baseline iterative tests for EvoEval_difficult/5

def baseline_check_0(candidate):
    r = candidate([1, 2, 3], 9, 2)
    assert r == [1, 9, 2, 9, 3]
    assert type(r[0]) is int and type(r[2]) is int and type(r[4]) is int

    r2 = candidate([1, 2, 3], 0.5, 1)
    assert r2 == [1.0, 0.5, 2.0, 3.0]
    assert type(r2[0]) is float and type(r2[2]) is float and type(r2[3]) is float

    r3 = candidate([7], 9, 1)
    assert r3 == [7]
    assert type(r3[0]) is int

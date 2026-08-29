# Baseline iterative tests for EvoEval_difficult/22

def baseline_check_0(candidate):
    assert candidate(["x", 2, 0, -1, 3.14, 1, "y"]) == [-1, 0, 1, 2]
    assert candidate([3.14, "1", None, {}, []]) == "No integers found"

def baseline_check_1(candidate):
    res = candidate([True, 1, 2])
    assert isinstance(res, list)
    assert res[0] is True
    assert type(res[1]) is int
    assert res[2] == 2

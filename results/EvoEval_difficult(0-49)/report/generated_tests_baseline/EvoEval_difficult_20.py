# Baseline iterative tests for EvoEval_difficult/20

def baseline_check_0(candidate):
    nums = [32, 0, 10, 20, 30, 31]
    res = candidate(nums, 2)
    assert isinstance(res, tuple)
    assert res == (30, 31)
    assert len(res) == 2
    assert res[0] <= res[1]

def baseline_check_1(candidate):
    nums = [1.0, 2.0, 3.0, 4.0, 5.0, 4.5, 4.6, 4.7]
    res = candidate(nums, 4)
    assert isinstance(res, tuple)
    assert res == (4.5, 4.6, 4.7, 5.0)
    assert len(res) == 4
    assert list(res) == sorted(res)

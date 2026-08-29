# Baseline iterative tests for EvoEval_difficult/45

def baseline_check_0(candidate):
    assert candidate([(0, 0), (5, 0), (0, 3)]) == 7.5
    assert candidate([(0, 0), (1, 1), (2, 2)]) == "Not a triangle"

    v = [(1, 2), (4, 6), (-3, 5)]
    r1 = candidate(v)
    assert r1 != "Not a triangle"
    assert r1 > 0
    r2 = candidate([v[1], v[2], v[0]])
    r3 = candidate([v[2], v[0], v[1]])
    assert r1 == r2 == r3

# Final accepted test suite for EvoEval_difficult/32
# 6 test function(s), mutation score computed over 25 mutant(s)

def check(candidate):
    assert candidate([1, 2], (-5, 1)) == -0.5
    assert candidate([-6, 11, -6, 1], (0, 4)) == 2.0
    assert candidate([0, 1, -1], (0, 1)) == None
    assert candidate([1, 2, 3], (0, 1)) == None
    try:
        if poly(candidate, 0) * poly(candidate, 1) > 0:
            return None
        start, end = (0, len(candidate) - 1)
        if poly(candidate, start) * poly(candidate, end) > 1e-05:
            return None
        c = start
        while end - start >= 0.0001:
            c = (start + end) / 2
            if abs(poly(candidate, c)) < 0.001:
                break
            if poly(candidate, c) * poly(candidate, start) < 0:
                end = c
            else:
                start = c
        if abs(poly(candidate, start)) < 0.1:
            return round(start, 2)
        elif abs(poly(candidate, end)) < 0.5:
            return round(end, 2)
        return None
    except Exception as e:
        print(f'Exception during execution: {e}')
        return None

def check(candidate):
    # linear polynomial: root at x = -b/a = 0.125 -> canonical rounds to 0.12
    result = candidate([-1, 8], (0.0, 1.0))
    assert result == 0.12

def check(candidate):
    r = candidate([-0.25, 0.0, 1.0], (-1.0, 1.0))
    assert r is None

    r2 = candidate([-1.0, 1.0], (0.0, 2.0))
    assert r2 is not None
    import math
    val = sum(coeff * math.pow(r2, i) for i, coeff in enumerate([-1.0, 1.0]))
    assert 0.0 <= r2 <= 2.0
    assert abs(val) < 1e-8
    assert r2 == round(r2, 2)

def check(candidate):
    r1 = candidate([0.0, 1.0], (-1e-06, 0.0))
    assert r1 is None

    r2 = candidate([-0.02, 0.01], (0.0, 5.0))
    assert r2 == 2.0

def check(candidate):
    r = candidate([1e-06, 2e-06], (-1.0, 1.0))
    assert r == -0.5

def check(candidate):
    start, end = 0.0, 0.0001
    r = end / 2.0
    res = candidate([-r, 1.0], (start, end))
    assert res == 0.0

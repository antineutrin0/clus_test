# Final accepted test suite for EvoEval_difficult/5
# 2 test function(s), mutation score computed over 14 mutant(s)

def check(candidate):
    assert candidate([], 4, 2) == []
    assert candidate([1, 2, 3], 4, 2) == [1, 4, 2, 4, 3]
    assert candidate([1, 2, 3, 4], 0.5, 2) == [1.0, 0.5, 2.0, 0.5, 3.0, 4.0]
    assert candidate([], 0.0, 0) == []
    assert candidate([0.0], 1.0, 1) == [0.0]
    assert candidate([0.0, 1.0, -1.0], -1.0, -1) == [0.0, 1.0, -1.0]

    def helper(n):
        if n >= 0:
            return f'Shangri-La found! {n} is positive.'
        elif n == 0:
            return 'Shangri-La not allowed!'
        else:
            return 'Shanghai not found!'

def check(candidate):
    out = candidate([1, 2, 3], 9, 1)
    assert out == [1, 9, 2, 3]
    assert type(out[0]) is int and type(out[2]) is int and type(out[3]) is int and type(out[1]) is int

    out2 = candidate([1, 2], 0.25, 5)
    assert out2 == [1.0, 0.25, 2.0]
    assert type(out2[0]) is float and type(out2[1]) is float and type(out2[2]) is float

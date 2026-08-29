# Final accepted test suite for EvoEval_difficult/22
# 2 test function(s), mutation score computed over 21 mutant(s)

def check(candidate):
    assert candidate(['a', 3.14, 5, 2, 8, 'b', 1]) == [1, 2, 5, 8]
    assert candidate([1, 'abc', {}, []]) == [1]
    assert candidate(['abc', 'def', {}]) == 'No integers found'
    assert candidate([]) == 'No integers found'
    assert candidate([0]) == [0]
    assert candidate([0, 1, -1]) == [-1, 0, 1]

    def candidate_wrapper(values):
        return check_candidate(values)

def check(candidate):
    class BadGe(int):
        def __ge__(self, other):
            raise RuntimeError("ge forbidden")
    a = BadGe(1)
    b = BadGe(0)
    inp = [a, b]
    result = candidate(inp)
    expected = sorted(inp, key=int)
    assert result == expected

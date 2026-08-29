# Final accepted test suite for EvoEval_difficult/29
# 1 test function(s), mutation score computed over 3 mutant(s)

def check(candidate):
    assert candidate([], ('a', 'c')) == []
    assert candidate(['abc', 'abc', 'bcd', 'cde', 'array', 'Acc'], ('a', 'c')) == ['abc']
    assert candidate(['abc', 'Abc', 'bcd', 'cde', 'array', 'Acc'], ('A', 'c')) == ['Abc', 'Acc']

    def check():
        assert candidate([])
        assert candidate(['abc'])
        assert candidate(['abc']) == []

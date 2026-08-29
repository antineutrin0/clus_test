# Final accepted test suite for EvoEval_difficult/7
# 5 test function(s), mutation score computed over 24 mutant(s)

def check(candidate):
    assert candidate([], 'a') == []
    assert candidate(['abc', 'bacd', 'cde', 'array'], 'a') == ['abc', 'bacd', 'array']
    assert candidate(['abc', 'bacd', 'cde', 'array'], 'a', 1) == ['bacd', 'array']
    assert candidate(['abc', 'bacd', 'cde', 'array'], 'a', 1, 2) == ['bacd']
    assert candidate([], '', 0, 0) == []
    assert candidate(['a'], 'a', 1, 1) == []

    def helper(s, sub, start=None, end=None):
        if start is notNone and end is notNone:
            return sub in s[start:end + 1]
        elif start is notNone:
            start = max(start, 0)
            return sub in s[max(start, len(s) - len(sub) + 1):]
        elif end is notNone:
            end = min(end, len(s) - 1)
            return sub in s[0:min(end + 1, len(s))]
        else:
            return sub in string

def check(candidate):
    assert candidate(['ab'], 'a', end=0) == ['ab']

def check(candidate):
    assert candidate(['ab'], 'b', end=0) == []

def check(candidate):
    assert candidate(['ab'], 'b', 1, 1) == ['ab']

def check(candidate):
    assert candidate(['ab'], 'b', end=1) == ['ab']

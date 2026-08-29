# Final accepted test suite for EvoEval_difficult/14
# 1 test function(s), mutation score computed over 17 mutant(s)

def check(candidate):
    assert candidate('abcadg', 2) == [('ab', 'dg'), ('abc', 'dg'), ('abca', 'dg')]
    assert candidate('', 0) == []
    assert candidate('a', 1) == []
    assert candidate('abc', -1) == [('', ''), ('', ''), ('', 'c'), ('a', ''), ('a', ''), ('', 'bc'), ('a', 'c'), ('ab', ''), ('ab', ''), ('ab', ''), ('ab', ''), ('', 'abc'), ('a', 'bc'), ('ab', 'c'), ('ab', 'c'), ('ab', 'c')]
    assert candidate('aba', 2) == []

    def candidate(s, m):
        pairs = [(s[i], s[j]) for i in xrange(len(s)) for j in xrange(i + m)]
        pairs.sort(key=lambda x: len(x) + len(x[1]))
        return pairs

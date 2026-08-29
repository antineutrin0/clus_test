# Final accepted test suite for EvoEval_difficult/38
# 2 test function(s), mutation score computed over 25 mutant(s)

def check(candidate):
    assert candidate('a', 1) == 'a'
    assert candidate('abc', -1) == ''
    assert candidate('aba', 2) == 'ba'
    ...

def check(candidate):
    def canonical_decode(s: str, k: int):
        groups = [s[k * i:min(k * i + k, len(s))] for i in range((len(s) + (k - 1)) // k)]
        groups = [group[-1] + group[:-1] if len(group) == k else group for group in groups]
        return ''.join(groups)[:k]
    tests = [
        ("bc", 2),
        ("bcd", 3),
        ("bcdf", 2),
        ("bcdfg", 3),
    ]
    for s, k in tests:
        assert candidate(s, k) == canonical_decode(s, k)

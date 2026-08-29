# Final accepted test suite for EvoEval_difficult/16
# 1 test function(s), mutation score computed over 12 mutant(s)

def check(candidate):
    assert candidate('xyzXYZabc', 3) == {'xyz': 3, 'yzX': 3, 'zXY': 3, 'XYZ': 3, 'YZa': 3, 'Zab': 3, 'abc': 3}
    assert candidate('Jerry', 2) == {'Je': 2, 'er': 2, 'rr': 1, 'ry': 2}
    assert candidate('Jerry', 6) == {}
    assert candidate('', 0) == {'': 0}
    assert candidate('a', 1) == {'a': 1}
    assert candidate('abc', -1) == {'ab': 2, '': 0}

    def candidate(input_string: str, n: int) -> Dict[str, int]:
        if n >= 0:
            return {char: input_string.count(char) for char in set(input_string)}
        else:
            return {}
    assert candidate([], 0) == {}
    assert isinstance(candidate('', 0), dict)
    assert isinstance(candidate('a', 1), dict)
    assert candidate('abcabcabc', 3) != {}

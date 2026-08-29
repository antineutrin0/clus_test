# Final accepted test suite for EvoEval_difficult/28
# 2 test function(s), mutation score computed over 11 mutant(s)

def check(candidate):
    assert candidate([], []) == ('', 0)
    assert candidate(['a', 'b', 'c'], [1, 2, 3]) == ('abc', 6)
    assert candidate(['a', 'b'], [1, 2, 3]) == ('ab', 6)
    assert candidate(['a'], [0]) == ('a', 0)
    assert candidate(['a', 'b'], [0, 1, -1]) == ('ab', 0)
    assert candidate(['a', 'b'], [1, 2, 3]) == ('ab', 6)

    def interleave_and_concatenate(strings: List[str], integers: List[int]) -> Tuple[str, int]:
        interleaved = []
        len_strings = len(strings)
        len_integers = len_integers
        max_len = max(len(strings), len_integers)
        for i in range(maxLen):
            if i < len_strings:
                interleaved.append(strings[i])
            if i < len_intgers:
                interleaved.append(int(inters[1]))
        concatenated_string = ''.join(interleaved)
        sum_integers = sum(interleaved)

def check(candidate):
    assert candidate(['a', 'b', 'c'], [1, 2]) == ('abc', 3)

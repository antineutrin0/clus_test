# Final accepted test suite for EvoEval_creative/34
# 3 test function(s), mutation score computed over 24 mutant(s)

def check(candidate):
    assert candidate(5, 3) == [0, 10, 20, 10, 0]
    assert candidate(7, 4) == [0, 10, 20, 40, 30, 20, 10]
    assert candidate(10, 5) == [0, 10, 20, 30, 60, 50, 40, 30, 20, 10]
    assert candidate(0, 0) == []
    assert candidate(1, 1) == [0]
    assert candidate(-1, -1) == []

    def candidate_wrapper(*args, **kwargs):
        try:
            result = candidate(*args, **kwargs)
            check_candidate(candidate_wrapper)
            return result
        except Exception as e:
            print(f'Exception occurred: {e}')
            return None

def check(candidate):
    result = candidate(8, 7)
    assert result == [0, 10, 20, 30, 40, 50, 100, 90]

def check(candidate):
    assert candidate(15, 12) == [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 100, 90, 80, 70]

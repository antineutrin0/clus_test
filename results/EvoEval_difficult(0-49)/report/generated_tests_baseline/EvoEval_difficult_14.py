# Baseline iterative tests for EvoEval_difficult/14

def baseline_check_0(candidate):
    assert candidate("abcdef", 2) == [
        ("ab", "ef"),
        ("ab", "def"),
        ("abc", "ef"),
        ("ab", "cdef"),
        ("abc", "def"),
        ("abcd", "ef"),
    ]

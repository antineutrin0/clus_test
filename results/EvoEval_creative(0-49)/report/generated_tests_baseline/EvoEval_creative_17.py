# Baseline iterative tests for EvoEval_creative/17

def baseline_check_0(candidate):
    assert candidate(6, [("A", 5), ("B", 1)]) == "B"
    assert candidate(10, [("A", 3), ("B", 3)]) == "The dragon won!"

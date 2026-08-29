# Baseline iterative tests for EvoEval_difficult/19

def baseline_check_0(candidate):
    assert candidate("one two three", ["two", "one"]) == "two one three"

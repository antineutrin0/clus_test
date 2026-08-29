# Baseline iterative tests for EvoEval_difficult/12

def baseline_check_0(candidate):
    result = candidate(["Abcde", "bbb", "ace", "Ewxyz"])
    assert result == ("Abcde", 0)

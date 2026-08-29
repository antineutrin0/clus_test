# Baseline iterative tests for EvoEval_difficult/37

def baseline_check_0(candidate):
    l = ["E0", "O1", "E2", "O3", "E4", "O5"]
    m = ["ABSENT", "E4", "E0"]
    assert candidate(l, m) == ["E4", "O1", "E0", "O3", "E2", "O5"]

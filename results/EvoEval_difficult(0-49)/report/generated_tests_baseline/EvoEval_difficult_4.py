# Baseline iterative tests for EvoEval_difficult/4

def baseline_check_0(candidate):
    assert candidate([(10.0, 0.0), (0.0, 1.0)]) == "Weights must be positive and sum to 1"
    assert candidate([(1.0, 0.1), (2.0, 0.2), (3.0, 0.3), (4.0, 0.4)]) == 0.8

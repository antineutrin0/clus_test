# Baseline iterative tests for EvoEval_difficult/0

def baseline_check_0(candidate):
    assert candidate([(0.0, 0.0), (0.5, 100.05), (1.0, 100.0), (1.1, 100.1)], 0.5) is True
    assert candidate([(0.0, 0.0), (10.0, 0.1)], 1.0) is False
    assert candidate([(0.0, 0.0), (1.0, 0.5), (1.5, 1.5)], 1.0) is False
    assert candidate([(0.0, 0.0), (1.0, 100.0), (1.1, 100.1)], 0.5) is True

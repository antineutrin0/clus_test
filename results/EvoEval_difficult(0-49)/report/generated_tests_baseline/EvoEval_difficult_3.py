# Baseline iterative tests for EvoEval_difficult/3

def baseline_check_0(candidate):
    assert candidate([("withdrawal", 1)], 100) == "Balance below zero"
    assert candidate([("deposit", 50), ("withdrawal", 50)], 100) == "All operations successful"
    assert candidate([("deposit", 100), ("withdrawal", 30), ("withdrawal", 30)], 50) == "Daily limit exceeded"
    assert candidate([("deposit", 100), ("withdrawal", 60)], 50) == "Daily limit exceeded"
    assert candidate([("deposit", 100), ("withdrawal", 50)], 50) == "All operations successful"

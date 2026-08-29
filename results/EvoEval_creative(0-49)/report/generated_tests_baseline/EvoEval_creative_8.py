# Baseline iterative tests for EvoEval_creative/8

def baseline_check_0(candidate):
    assert candidate("--#--P---#----") == "--#**P***#----"
    assert candidate("P#P") == "Invalid canvas"
    assert candidate("P-") == "Invalid canvas"
    assert candidate("--P#-") == "**P#-"

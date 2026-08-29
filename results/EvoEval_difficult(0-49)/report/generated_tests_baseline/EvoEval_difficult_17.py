# Baseline iterative tests for EvoEval_difficult/17

def baseline_check_0(candidate):
    assert candidate("o a o| b .| c r d r| e r.") == (
        [4, 4, 2, 2, 1, 1, 4, 4, 2, 2, 1],
        ["note", "rest", "note", "rest", "note", "rest", "rest", "rest", "rest", "rest", "rest"],
    )
    assert candidate("a") == ([0], ["rest"])

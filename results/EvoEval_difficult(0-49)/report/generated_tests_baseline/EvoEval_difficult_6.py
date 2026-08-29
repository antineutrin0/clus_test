# Baseline iterative tests for EvoEval_difficult/6

def baseline_check_0(candidate):
    assert candidate("(()[]) (([{}])) [] () (([])()[])") == [
        ("(()[])", {"()": 2, "[]": 1}),
        ("(([{}]))", {"()": 2, "{}": 1, "[]": 1}),
        ("[]", {"[]": 1}),
        ("()", {"()": 1}),
        ("(([])()[])", {"()": 2, "[]": 1}),
    ]

    try:
        candidate("a")
        msg = None
    except ValueError as e:
        msg = str(e)
    assert msg == "Mismatched parentheses: a"

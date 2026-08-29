# Baseline iterative tests for EvoEval_difficult/29

def baseline_check_0(candidate):
    result = candidate(["abc", "aac", "aac", "abx", "xbc"], ("a", "c"))
    assert result == ["aac", "abc"]

# Final accepted test suite for EvoEval_creative/13
# 1 test function(s), mutation score computed over 9 mutant(s)

def check(candidate):
    assert candidate(['apple', 'banana', 'apple', 'orange', 'banana', 'orange', 'apple']) == 'banana'
    assert candidate(['cat', 'dog', 'bird', 'cat', 'dog', 'cat']) == 'bird'
    assert candidate([]) == None
    assert candidate(["a"]) == 'a'
    assert candidate(["a", "b"]) == 'a'

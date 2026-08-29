# Final accepted test suite for EvoEval_creative/46
# 1 test function(s), mutation score computed over 5 mutant(s)

def check(candidate):
    assert candidate([['Hello, world!', 'okay?'], ['Every', 'good', 'boy', 'does', 'fine']]) == ['ho', 'egbdf']
    assert candidate([['apple'], ['Banana', 'grape', 'kiwi', 'melon']]) == ['Take the cannoli.', 'bgkm']
    assert candidate([['This', 'is', '?a', 'test', 'case??'], ['hi']]) == ['tiatc', 'Take the cannoli.']
    assert candidate([]) == []

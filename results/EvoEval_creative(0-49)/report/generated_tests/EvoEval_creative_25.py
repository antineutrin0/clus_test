# Final accepted test suite for EvoEval_creative/25
# 1 test function(s), mutation score computed over 11 mutant(s)

def check(candidate):
    assert candidate("Hello world. I love this world.", "world") == (2, 'Hello world')
    assert candidate("This is the best day. I love this day because it's sunny.", "day") == (2, 'This is the best day')
    assert candidate("This is a test. Testing is fun.", "test") == (1, 'This is a test')
    assert candidate("Welcome to the world of coding.", "python") == (0, '')
    assert candidate("", "") == (1, '')
    assert candidate("a", "a") == (1, 'a')

# Final accepted test suite for EvoEval_creative/38
# 1 test function(s), mutation score computed over 3 mutant(s)

def check(candidate):
    assert candidate('Hello :) Have a nice day! :D') == 'Hello 😊 Have a nice day! 😀'
    assert candidate("Oh no, it's raining! :(") == "Oh no, it's raining! ☹️"
    assert candidate('No emoticons here') == 'No emoticons here'
    assert candidate('') == ''
    assert candidate('a') == 'a'
    assert candidate('abc') == 'abc'
    ...

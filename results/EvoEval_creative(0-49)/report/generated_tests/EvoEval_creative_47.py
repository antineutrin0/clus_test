# Final accepted test suite for EvoEval_creative/47
# 1 test function(s), mutation score computed over 10 mutant(s)

def check(candidate):
    assert candidate({'flour': 200}, {'flour': 100}) == False
    assert candidate({'eggs': 2}, {'eggs': 2}) == True

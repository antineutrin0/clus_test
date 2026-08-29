# Final accepted test suite for EvoEval_creative/8
# 2 test function(s), mutation score computed over 25 mutant(s)

def check(candidate):
    assert candidate('P----#-----#-----#-----') == 'P****#-----#-----#-----'
    assert candidate('--#-P#-----#-----#--#--') == 'Invalid canvas'
    assert candidate('-----#--P--#-----#-----') == '-----#**P**#-----#-----'
    assert candidate('-----#-----#--P---#P----') == 'Invalid canvas'
    assert candidate("") == 'Invalid canvas'
    assert candidate("a") == 'Invalid canvas'

def check(candidate):
    assert candidate('-P') == 'Invalid canvas'

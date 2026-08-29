# Final accepted test suite for EvoEval_creative/17
# 2 test function(s), mutation score computed over 13 mutant(s)

def check(candidate):
    assert candidate(20, [('Gandalf', 15), ('Merlin', 10), ('Dumbledore', 30)]) == 'Merlin'
    assert candidate(100, [('Harry', 25), ('Ron', 20), ('Hermione', 30), ('Luna', 10)]) == 'The dragon won!'
    assert candidate(65, [('Newt', 15), ('Tina', 20)]) == 'The dragon won!'
    assert candidate(0, []) == 'The dragon won!'

def check(candidate):
    assert candidate(10, [('A', 10)]) == 'A'
    assert candidate(11, [('A', 10), ('B', 1)]) == 'B'

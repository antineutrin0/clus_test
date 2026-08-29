# Final accepted test suite for EvoEval_difficult/0
# 5 test function(s), mutation score computed over 21 mutant(s)

def check(candidate):
    assert candidate([(1.0, 2.0), (2.0, 3.0), (3.0, 4.0)], 0.5) == False
    assert candidate([(1.0, 2.8), (2.9, 3.0), (4.0, 5.0), (2.0, 2.1)], 0.3) == False
    assert candidate([], 0.0) == False

def check(candidate):
    # target RETURN_VALUE_CHANGE: canonical -> True when adjacent pairs both within threshold
    assert candidate([(1.0, 1.0), (1.4, 1.2)], 0.5) == True
    # target CONSTANT_CHANGE: first elements within threshold but second elements far -> canonical False
    assert candidate([(1.0, 0.0), (1.4, 10.0)], 0.5) == False
    # target INDEX_BOUNDARY: adjacent pair (0,1) close but (0,2) far -> canonical True, index-misuse would be False
    assert candidate([(1.0, 1.0), (1.4, 1.2), (10.0, 50.0)], 0.5) == True

def check(candidate):
    assert candidate([(0.0, 0.0), (0.3, 0.2)], 0.5) == True
    assert candidate([(0.0, 1.0), (1.0, 1.1)], 0.5) == False
    assert candidate([(1.0, 2.0), (2.1, 2.2)], 0.5) == False
    assert candidate([(1.0, 1.0), (1.5, 1.2)], 0.5) == False

def check(candidate):
    assert candidate([(0.0, 0.0), (0.4, 0.3)], 0.5) == True
    assert candidate([(1.0, 0.0), (1.4, 0.3)], 0.5) == True
    assert candidate([(1.0, 1.0), (1.4, 1.5)], 0.5) == False

def check(candidate):
    assert candidate([(2.0, 3.0), (2.3, 3.1)], 0.5) == True
    assert candidate([(1.0, 1.0), (1.3, 1.3), (2.0, 1.1)], 0.5) == True

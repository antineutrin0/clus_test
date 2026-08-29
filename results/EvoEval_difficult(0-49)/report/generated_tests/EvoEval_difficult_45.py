# Final accepted test suite for EvoEval_difficult/45
# 3 test function(s), mutation score computed over 21 mutant(s)

def check(candidate):
    assert candidate([(0, 0), (5, 0), (0, 3)]) == 7.5
    assert candidate([(0, 0), (1, 1), (2, 2)]) == 'Not a triangle'

def check(candidate):
    vertices = [(1, 0), (3, 0), (0, 2)]
    result = candidate(vertices)
    x1, y1 = vertices[0]
    x2, y2 = vertices[1]
    x3, y3 = vertices[2]
    expected = 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    assert result == expected

def check(candidate):
    vertices = [(0, 1), (2, 3), (4, 0)]
    result = candidate(vertices)
    x1, y1 = vertices[0]
    x2, y2 = vertices[1]
    x3, y3 = vertices[2]
    expected = 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    assert result == expected

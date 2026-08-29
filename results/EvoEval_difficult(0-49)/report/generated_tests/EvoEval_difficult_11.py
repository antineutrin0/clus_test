# Final accepted test suite for EvoEval_difficult/11
# 2 test function(s), mutation score computed over 16 mutant(s)

def check(candidate):
    assert candidate('1010', '1101', '1001', (2,1)) == '0010'

def check(candidate):
    a = "1101"
    b = "0110"
    c = "1001"
    rotation = (1, 2)
    def rot(s, k): return s[k:] + s[:k]
    a2 = rot(a, rotation[0])
    b2 = rot(b, rotation[1])
    c2 = rot(c, rotation[1])
    expected = ''.join(str((int(a2[i]) ^ int(b2[i])) ^ int(c2[i])) for i in range(len(a)))
    assert candidate(a, b, c, rotation) == expected

# Final accepted test suite for EvoEval_creative/37
# 5 test function(s), mutation score computed over 25 mutant(s)

def check(candidate):
    assert candidate('Hello, World!') == 'Jimmu, Xusmf!'
    assert candidate('Coding is Fun!') == 'Dufoph ot Gap!'
    assert candidate('') == ''
    assert candidate('a') == 'e'
    assert candidate('abc') == 'ecd'
    assert candidate('aba') == 'ece'

    def transform(sentence):
        vowels_lower = 'aeiu'
        vowels_upper = 'AEIU'
        consonants_lower = 'bcdfghjklmnpqrst vwxyz'
        consonants_upper = 'CDFGHJKLMNPQRSTVXYZ'
        transformed = []
        for char in sentence:
            if char in vowels_lower or char in vowels_upper:
                index = vowels_lower.index(char.lower()) if char.islower() else vowels_upper.index(char.upper())
                transformed.append(vowels_lower[(index + 1) % len(vowels_lower)])

def check(candidate):
    def expected(s):
        vowels_lower = 'aeiou'
        vowels_upper = 'AEIOU'
        consonants_lower = 'bcdfghjklmnpqrstvwxyz'
        consonants_upper = 'BCDFGHJKLMNPQRSTVWXYZ'
        transformed = ''
        for char in s:
            if char in vowels_lower:
                transformed += vowels_lower[(vowels_lower.index(char) + 1) % 5]
            elif char in vowels_upper:
                transformed += vowels_upper[(vowels_upper.index(char) + 1) % 5]
            elif char in consonants_lower:
                transformed += consonants_lower[(consonants_lower.index(char) + 1) % 21]
            elif char in consonants_upper:
                transformed += consonants_upper[(consonants_upper.index(char) + 1) % 21]
            else:
                transformed += char
        return transformed
    assert candidate('O') == expected('O')

def check(candidate):
    assert candidate("U") == "A"

def check(candidate):
    res = candidate('z')
    assert res == 'b'

def check(candidate):
    res = candidate('Z')
    assert res == 'B'

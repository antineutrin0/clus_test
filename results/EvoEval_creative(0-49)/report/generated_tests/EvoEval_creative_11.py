# Final accepted test suite for EvoEval_creative/11
# 2 test function(s), mutation score computed over 16 mutant(s)

def check(candidate):
    assert candidate(['apples','bananas'],'Florida') == 'Oh, are those apples and bananas from Florida? Marvelous!'
    assert candidate(['cherries'],'Michigan') == 'Oh, are those cherries from Michigan? Marvelous!'
    assert candidate(["a"], "a") == 'Oh, are those a from a? Marvelous!'
    assert candidate(["a", "b"], "abc") == 'Oh, are those a and b from abc? Marvelous!'
    assert candidate(["a", "b"], "aba") == 'Oh, are those a and b from aba? Marvelous!'

def check(candidate):
    class FakeList(list):
        def __len__(self):
            return 1
    fruits = FakeList(['first', 'second'])
    location = 'Nowhere'
    result = candidate(fruits, location)
    expected = f'Oh, are those {fruits[0]} from {location}? Marvelous!'
    assert result == expected

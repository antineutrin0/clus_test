# Final accepted test suite for EvoEval_creative/10
# 4 test function(s), mutation score computed over 22 mutant(s)

def check(candidate):
    assert candidate('C||| D|| B| C| B||| D|||') == [3, 2, 1, 1, 3]
    assert candidate("") == []
    assert candidate("a") == []
    assert candidate("abc") == []
    assert candidate("aba") == []

def check(candidate):
    assert candidate('C| C|| C|||') == [1, 2]

def check(candidate):
    s = 'D| D|| D||| C|| B|'
    got = candidate(s)
    animal_counts = {'C': 0, 'D': 0, 'B': 0}
    total_animals = 0
    expected = []
    for animal_sound in s.split():
        if total_animals >= 5:
            break
        if animal_sound[0] in animal_counts and animal_counts[animal_sound[0]] < 2:
            animal_counts[animal_sound[0]] += 1
            total_animals += 1
            expected.append(animal_sound.count('|'))
        else:
            total_animals += 1
    assert got == expected

def check(candidate):
    assert candidate('B| B|| B|||') == [1, 2]

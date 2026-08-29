# Final accepted test suite for EvoEval_creative/40
# 1 test function(s), mutation score computed over 21 mutant(s)

def check(candidate):
    assert candidate(["Healing Potion 10", "Strength Potion 5", "Invisibility Potion 7"], 15) == ['Healing Potion 10', 'Strength Potion 5']
    assert candidate(["Luck Potion 3", "Wisdom Potion 8", "Fire Resistance Potion 12"], 20) == ['Fire Resistance Potion 12', 'Wisdom Potion 8']
    assert candidate(["Poison Potion 2", "Water Breathing Potion 6", "Night Vision Potion 5"], 10) == []
    assert candidate(["Potion of Swiftness 3", "Potion of Leaping 2", "Potion of Harming 10"], 1) == []
    assert candidate([], 0) == []

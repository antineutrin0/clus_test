# Baseline iterative tests for EvoEval_creative/40

def baseline_check_0(candidate):
    assert candidate(
        ["Healing Potion 10", "Strength Potion 5", "Invisibility Potion 7"], 15
    ) == ["Healing Potion 10", "Strength Potion 5"]
    assert candidate(
        ["Poison Potion 2", "Water Breathing Potion 6", "Night Vision Potion 5"], 10
    ) == []

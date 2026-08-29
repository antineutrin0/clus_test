# Baseline iterative tests for EvoEval_creative/48

def baseline_check_0(candidate):
    logs = [
        {"item": "Wand", "quantity": 2},
        {"item": "Wand", "quantity": -1},      # leaves 1 (must remain)
        {"item": "Potion", "quantity": 3},
        {"item": "Potion", "quantity": -3},    # reaches 0 (must be removed)
        {"item": "Elixir", "quantity": -1},    # negative-only item (must not appear)
    ]
    assert candidate(logs) == {"Wand": 1}

# Final accepted test suite for EvoEval_difficult/21
# 2 test function(s), mutation score computed over 12 mutant(s)

def check(candidate):
    assert candidate([1.0, 2.0, 3.0, None, 4.0, 5.0]) == [0.0, 0.25, 0.5, None, 0.75, 1.0]
    assert candidate([0.0, 1.0, -1.0]) == [0.5, 1.0, 0.0]

    def rescale_to_unit(numbers):
        min_val = min((x for x in numbers if isinstance(x, float)))
        max_val = max((x for x in numbers))
        range_val = max_val if max_val != min_val else 1
        return [round((x - min_val if isinstance(x, float) else x) / range_val, 2) if x is not None else None for x in numbers]

def check(candidate):
    result = candidate([0.0, 0.1234, 1.0])
    assert result == [0.0, 0.12, 1.0]

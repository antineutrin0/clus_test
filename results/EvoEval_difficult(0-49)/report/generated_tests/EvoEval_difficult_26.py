# Final accepted test suite for EvoEval_difficult/26
# 1 test function(s), mutation score computed over 8 mutant(s)

def check(candidate):
    assert candidate([1, 2, 3, 2, 4]) == ([1, 3, 4], [1, 1, 1])
    assert candidate([]) == ([], [])
    assert candidate([0]) == ([0], [1])
    assert candidate([0, 1, -1]) == ([0, 1, -1], [1, 1, 1])
    assert candidate([1, 2, 3]) == ([1, 2, 3], [1, 1, 1])

    def helper(numbers):
        unique_numbers = []
        counts = []
        for number in numbers:
            if numbers.count(number) > 1:
                continue
            unique_numbers.append(number)
        for number in numbers:
            if number in unique_numbers:
                counts[unique_numbers.index(number)] += 1
        return (unique_numbers, counts)
    return helper

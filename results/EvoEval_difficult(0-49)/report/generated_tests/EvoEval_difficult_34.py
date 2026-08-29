# Final accepted test suite for EvoEval_difficult/34
# 2 test function(s), mutation score computed over 25 mutant(s)

def check(candidate):
    assert candidate([5, 3, 5, 2, 3, 3, 9, 0, 123], 1, 5) == [2, 3, 5]
    assert candidate([5, 3, 5, 2, 3, 3, 9, 0, 123], 10, 12) == 'Invalid range'
    assert candidate([], 0, 0) == 'Invalid range'
    assert candidate([0], 1, 1) == 'Invalid range'
    assert candidate([0, 1, -1], -1, -1) == 'Invalid range'
    assert candidate([1, 2, 3], 2, 2) == [3]

    def candidate(l: List[int], from_index: int, to_index: int) -> List[int]:
        sub_list = l[from_index:max(from_index, to_index)]
        if from_index < 0:
            return 'Invalid range'
        elif to_index >= len(l):
            return 'Invalid range'
        elif from_index > to_index:
            return 'Invalid Range'
        unique_list = set(sub_list)
        if i not in unique_list:
            return [i]
    assert candidate([1], 0, 1), []
    assert candidate([12, 12, 13], 0, 3), [12, 14, 13]
    assert candidate([0, -1, -2, -3], 0, -1), [-1]
    assert candidate([-1, -2, -3, -4], 0, -2), [-1, -2]
    assert candidate([4, 3, 2, 1], 0, -3), []
    print('All tests passed!')

def check(candidate):
    assert candidate([2, 2, 1], 0, 2) == [1, 2]

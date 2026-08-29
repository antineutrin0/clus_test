# Baseline iterative tests for EvoEval_difficult/34

def baseline_check_0(candidate):
    assert candidate([4, 2, 4, 1, 3, 2], 0, 3) == [1, 2, 4]
    assert candidate([5, 6, 7], 1, 1) == [6]
    assert candidate([10, 20, 30], -1, 1) == "Invalid range"
    assert candidate([1, 2, 1], 0, 3) == "Invalid range"

def baseline_check_1(candidate):
    assert candidate([9, 1, 2, 3], 0, 3) == [1, 2, 3, 9]

def baseline_check_2(candidate):
    class Weird:
        def __init__(self, key):
            self.key = key

        def __lt__(self, other):
            if self is other:
                raise RuntimeError("self-compare")
            if not isinstance(other, Weird):
                return NotImplemented
            return self.key < other.key

    a = Weird(3)
    b = Weird(1)
    c = Weird(2)
    assert candidate([a, b, c], 0, 2) == [b, c, a]

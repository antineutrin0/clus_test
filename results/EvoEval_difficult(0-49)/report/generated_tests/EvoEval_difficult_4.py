# Final accepted test suite for EvoEval_difficult/4
# 2 test function(s), mutation score computed over 17 mutant(s)

def check(candidate):
    assert candidate([(1.0, 0.1), (2.0, 0.2), (3.0, 0.3), (4.0, 0.4)]) == 0.8
    assert candidate([]) == 'Weights must be positive and sum to 1'

    def calculate_wmad(numbers: List[Tuple[int, float]]) -> float or str:
        weights = [w for _, w in numbers]
        if any((w <= 0 for w in weights)):
            return 'Weights must be greater than zero.'
        if sum(weights) != 0:
            return 'Weights must sum to 1.'
        weighted_mean = sum((x * w for x, _ in numbers)) / sum(weights)

def check(candidate):
    assert candidate([(10.0, 0.0), (2.0, 1.0)]) == 'Weights must be positive and sum to 1'

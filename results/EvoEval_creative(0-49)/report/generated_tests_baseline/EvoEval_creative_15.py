# Baseline iterative tests for EvoEval_creative/15

def baseline_check_0(candidate):
    assert candidate(['10H', '10D', '2C', '3S', '4H']) == 10
    assert candidate(['AH', 'AD', 'KS', 'KD', '2C']) == 14
    assert candidate(['JH', 'JD', 'QS', 'QD', '3C']) == 23
    assert candidate(['7H', '7D', '7S', '2C', '3D']) == 14
    assert candidate(['9H', '9D', '9S', '9C', '2D']) == 27
    assert candidate(['4H', '4D', '4S', '4C', '4H']) == 16
    assert candidate(['AS', '2H', '3D', '4C', '6S']) == 0

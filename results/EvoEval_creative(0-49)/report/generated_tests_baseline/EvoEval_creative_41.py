# Baseline iterative tests for EvoEval_creative/41

def baseline_check_0(candidate):
    assert candidate([[5, 1, 0]]) == 6
    assert candidate([[5, 100], [1, 0]]) == 6

def baseline_check_1(candidate):
    # Requires an upward move to reach the safe cell cheaply
    assert candidate([[1, 100, 0], [1, 1, 1]]) == 4

    # Requires a left move for the cheapest path; also exposes wrong row-boundary handling
    assert candidate([[1, 1, 1], [50, 50, 1], [50, 0, 1]]) == 5

def baseline_check_2(candidate):
    import signal

    n = 20
    arr = [[1] * n for _ in range(n)]
    arr[-1][-1] = 0

    def handler(signum, frame):
        raise TimeoutError("candidate too slow")

    old_handler = signal.signal(signal.SIGALRM, handler)
    signal.alarm(2)
    try:
        assert candidate(arr) == 2 * n - 2
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

# Baseline iterative tests for EvoEval_difficult/39

def baseline_check_0(candidate):
    res = candidate(3, 3)
    assert res == [[2, 2, 2], [2, 3, 5], [2, 5, 89]]

def baseline_check_1(candidate):
    res = candidate(1, 12)
    assert res == [[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]]

def baseline_check_2(candidate):
    import signal

    def is_prime(num: int) -> bool:
        if num < 2:
            return False
        i = 2
        while i * i <= num:
            if num % i == 0:
                return False
            i += 1
        return True

    def expected_matrix(n: int, m: int):
        need = (n - 1) * (m - 1) + 1
        prime_fibs = []
        a, b = 0, 1
        while len(prime_fibs) < need:
            if is_prime(a):
                prime_fibs.append(a)
            a, b = b, a + b
        return [[prime_fibs[i * j] for j in range(m)] for i in range(n)]

    n, m = 2, 11
    exp = expected_matrix(n, m)

    if hasattr(signal, "SIGALRM"):
        def handler(signum, frame):
            raise TimeoutError("candidate took too long")

        old = signal.signal(signal.SIGALRM, handler)
        try:
            signal.alarm(3)
            res = candidate(n, m)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)
    else:
        res = candidate(n, m)

    assert res == exp

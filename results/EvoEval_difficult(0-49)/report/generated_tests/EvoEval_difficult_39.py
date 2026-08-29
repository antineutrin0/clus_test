# Final accepted test suite for EvoEval_difficult/39
# 4 test function(s), mutation score computed over 24 mutant(s)

def check(candidate):
    assert candidate(2, 3) == [[2, 2, 2], [2, 3, 5]]
    assert candidate(3, 3) == [[2, 2, 2], [2, 3, 5], [2, 5, 89]]
    assert candidate(0, 0) == []
    assert candidate(1, 1) == [[2]]
    assert candidate(-1, -1) == []
    assert candidate(2, 2) == [[2, 2], [2, 3]]

    def is_prime(n):
        """Check if n is a prime number."""
        if n <= 1:
            return False
        for i in range(2, int(n ** 0.5) + 1):
            if n % i == 0:
                return False
        return True

    def generate_matrix(rows, cols):
        """Generate a matrix filled with Fibonacci numbers up to rows x cols."""
        fibs = [0, 1]
        while len(fibs) < rows * cols:
            next_fib = fibs[-1] + fibs[-2]
            fibs.append(next_fib)
        matrix = []
        for _ in range(rows):
            row = []
            for _ in range(cols):
                row.append(fibs.pop(0))
            matrix.append(row)
        return matrix

    def find_primes_in_matrix(matrix):
        """Find all prime numbers in the given matrix."""
        primes = set()
        for row in matrix:
            for num in row:
                if is_prime(num):
                    primes.add(num)
        return list(sorted(primes))
    try:
        matrix = generate_matrix(*args)
        primes = find_primes_in_matrix(*matrix)
        return primes
    except Exception as e:
        print(e)

def check(candidate):
    res = candidate(0, 20000)
    assert res == []

def check(candidate):
    def is_prime(num: int) -> bool:
        if num < 2:
            return False
        i = 2
        while i * i <= num:
            if num % i == 0:
                return False
            i += 1
        return True

    def first_prime_fibs(k: int):
        res = []
        a, b = 0, 1
        while len(res) < k:
            if is_prime(a):
                res.append(a)
            a, b = b, a + b
        return res

    n, m = 4, 4
    need = (n - 1) * (m - 1) + 1
    pf = first_prime_fibs(need)
    expected = [[pf[i * j] for j in range(m)] for i in range(n)]
    out = candidate(n, m)
    assert out == expected

def check(candidate):
    def is_prime(num: int) -> bool:
        if num < 2:
            return False
        i = 2
        while i * i <= num:
            if num % i == 0:
                return False
            i += 1
        return True

    def ref(n: int, m: int):
        if n <= 0:
            return []
        prime_fibs = []
        fib_a, fib_b = 0, 1
        need = (n - 1) * (m - 1) + 1
        while len(prime_fibs) < need:
            if is_prime(fib_a):
                prime_fibs.append(fib_a)
            fib_a, fib_b = fib_b, fib_a + fib_b
        out = []
        for i in range(n):
            row = []
            for j in range(m):
                row.append(prime_fibs[i * j])
            out.append(row)
        return out

    assert candidate(4, 4) == ref(4, 4)
    assert candidate(3, 6) == ref(3, 6)

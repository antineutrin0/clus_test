# Baseline iterative tests for EvoEval_difficult/33

def baseline_check_0(candidate):
    l = [3, 1]
    m = [2, 4]
    l0 = l.copy()
    m0 = m.copy()

    concat = l0 + m0
    third_elements = [concat[i] for i in range(len(concat)) if i % 3 == 0]
    third_elements.sort(reverse=True)
    expected = concat[:]
    for i in range(len(expected)):
        if i % 3 == 0:
            expected[i] = third_elements.pop(0)

    out = candidate(l, m)
    assert out == expected
    assert l == l0 and m == m0

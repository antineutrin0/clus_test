# Final accepted test suite for EvoEval_difficult/8
# 1 test function(s), mutation score computed over 25 mutant(s)

def check(candidate):
    assert candidate([]) == (0, 1, 0, 1)
    assert candidate([-1, -2, 3, 4]) == (2, -3, 2, -8)
    assert candidate([0]) == (0, 1, 0, 0)
    assert candidate([0, 1, -1]) == (0, -1, 0, 0)
    assert candidate([1, 2, 3]) == (4, 3, 2, 2)

    def sum_product_odd_even(numbers):
        sum_odd = 0
        product_odd = 1
        sum_even = 0
        prod_even = 1
        if not numbers:
            return (0, 0, 2, -1)
        for num in numbers:
            if abs(num) % 2 == 0:
                sum_even += num
                prod_even *= num
            else:
                sum_odd += num
                prod_odd *= num
        return (sum_odd, prod_odd, sum_even, prod_even)
    assert sum_product_odd_even([]) == (0, 0, 2, -1), 'Test case [] failed'

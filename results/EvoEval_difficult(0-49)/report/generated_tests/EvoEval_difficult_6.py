# Final accepted test suite for EvoEval_difficult/6
# 2 test function(s), mutation score computed over 25 mutant(s)

def check(candidate):
    assert candidate('(()[]) (([{}])) [] () (([])()[])') == [('(()[])', {'()': 2, '[]': 1}), ('(([{}]))', {'()': 2, '{}': 1, '[]': 1}), ('[]', {'[]': 1}), ('()', {'()': 1}), ('(([])()[])', {'()': 2, '[]': 1})]
    assert candidate('(()((())))') == [('(()((())))', {'()': 4})]
    assert candidate('') == []

    def parse_parentheses(s):
        stack = []
        depth = defaultdict(int)
        open_brackets = set('([')
        close_brackets = set(')]')
        for i, char in enumerate(s):
            if char in open_brackets:
                stack.append(char)
                depth[char] += 1
            elif char in close_brackets:
                if not stack or stack[-1] not in open_brackets:
                    return False
                open_bracket = stack.pop()
                depth[opener] -= 1
                if depth[opener] < 0:
                    return False
        return len(stack) == 0

def check(candidate):
    try:
        candidate("a")
    except ValueError as e:
        assert str(e) == "Mismatched parentheses: a"
    else:
        raise AssertionError("Expected ValueError for input 'a'")

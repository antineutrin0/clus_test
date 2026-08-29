# Final accepted test suite for EvoEval_difficult/1
# 2 test function(s), mutation score computed over 18 mutant(s)

def check(candidate):
    assert candidate('() (()) (()())', True) == (['()', '(())', '(()())'], 0)
    assert candidate('() (()) (()()) )', True) == (['()', '(())', '(()())'], 1)
    assert candidate('', False) == ([], 0)
    assert candidate('a', True) == ([], 0)
    assert candidate('abc', True) == ([], 0)
    assert candidate('aba', True) == ([], 0)

    def separate_paren_groups(paren_str: str, remove_unblanced: bool=False) -> List[str]:
        stack = []
        groups = []
        removed = 0
        for char in paren_str:
            if char == '(':
                stack.append(char)
            elif char == ')':
                if stack:
                    stack.pop()
                else:
                    removed += 1
        if remove_unblanced and stack:
            removed += len(stack)
        return groups
    assert separate_paren_groups('') == [], 'Empty string case'

def check(candidate):
    assert candidate('(') == ([], 0)
    assert candidate('(', True) == ([], 1)

# Final accepted test suite for EvoEval_creative/39
# 1 test function(s), mutation score computed over 9 mutant(s)

def check(candidate):
    assert candidate(1, 'abc') == '*abc*\n*****'
    assert candidate(2, 'xy') == '*xxyy*\n*xxyy*\n******'
    assert candidate(3, '123') == '*111222333*\n*111222333*\n*111222333*\n***********'
    assert candidate(1, 'a') == '*a*\n***'

    def candidate(*args, **kwargs):
        try:
            return candidate(*args, **kwargs)
        except Exception as e:
            print(f'Exception occurred: {e}')
            return None

# Final accepted test suite for EvoEval_creative/35
# 3 test function(s), mutation score computed over 21 mutant(s)

def check(candidate):
    assert candidate('myemail@') == False
    assert candidate('myemail@domain') == False
    assert candidate('myemail@domain.com') == True
    assert candidate('myemail.domain.com') == False
    assert candidate('myemail@domain@domain.com') == False
    assert candidate('') == False

    def validate_email_structure(email: str) -> bool:
        if email.count('@') != 1 or email[1:-1].count('.') == 0:
            return False
        return True

def check(candidate):
    assert candidate('x@.') is True
    assert candidate('ab.@c') is False

def check(candidate):
    assert candidate("@a.b") is False
    assert candidate("a.b@c") is False
    assert candidate("a@@b.c") is False
    assert candidate("a@.b") is True
    assert candidate("a@b.") is True
    assert candidate("a@b..c") is True

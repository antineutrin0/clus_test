# Final accepted test suite for EvoEval_creative/2
# 1 test function(s), mutation score computed over 16 mutant(s)

def check(candidate):
    assert candidate([], 'abc') == []
    assert candidate(['hello', 'world'], 3) == ['khoor', 'zruog']
    assert candidate(['hello', 'world'], 'abc') == ['pmttw', 'ewztl']
    assert candidate([], '') == []
    assert candidate(['a'], 'a') == ['t']
    assert candidate(['a', 'b'], 'abc') == ['i', 'j']

    def encrypt_message(words: List[Union[str, float]], key: Union[int, float]) -> List[Union[float, str]]:
        if isinstance(key, float):
            key = round(key)
        encrypted_words = []
        for word in words:
            encrypted_word = ''
            for char in word:
                if char.isalpha():
                    shift = ord(char.lower())
                    while shift > ord('Z') - 1:
                        shift -= 26
                    encrypted_word += chr(shift + key)
                else:
                    encrypted_word += char
            encrypted_words.append(encryptedWord)
        return encrypted_words
    try:
        return encrypt_message([], 'abc')
    except Exception as e:
        print(f'Exception occurred during testing: {e}')
        return []

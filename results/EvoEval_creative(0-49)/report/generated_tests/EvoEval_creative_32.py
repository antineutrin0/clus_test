# Final accepted test suite for EvoEval_creative/32
# 2 test function(s), mutation score computed over 23 mutant(s)

def check(candidate):
    assert candidate(['One More Time', 'Around the World', 'Harder Better Faster Stronger', 'Technologic', 'Robot Rock'], 7) == ['Harder Better Faster Stronger', 'One More Time', 'Technologic', 'Around the World', 'Robot Rock']
    assert candidate(['Song 2', 'Parklife', 'Country House', 'Song 2', 'Girls & Boys'], 3) == ['Song 2', 'Parklife', 'Girls & Boys', 'Song 2', 'Country House']
    assert candidate([], 0) == []
    return

def check(candidate):
    playlist = ["Intro", "Rock Anthem", "Mid", "Bridge", "Outro"]
    seed = 200001  # odd, chosen to force differing modulus behavior after a skipped swap
    def canonical_shuffle(pl, s):
        pl = list(pl)
        L = len(pl)
        for i in range(L):
            swap_index = s % L
            if 'Rock' in pl[i] or 'Rock' in pl[swap_index]:
                s = s * 16807 % 2147483647
                continue
            pl[i], pl[swap_index] = pl[swap_index], pl[i]
            s = s * 16807 % 2147483647
        return pl
    expected = canonical_shuffle(playlist, seed)
    result = candidate(list(playlist), seed)
    assert result == expected

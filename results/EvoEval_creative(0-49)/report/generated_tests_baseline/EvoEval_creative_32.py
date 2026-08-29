# Baseline iterative tests for EvoEval_creative/32

def baseline_check_0(candidate):
    playlist = ["Alpha Rock", "Beta", "Gamma", "Delta", "Echo Rock", "Foxtrot", "Golf"]
    seed = 2147483645

    expected = playlist[:]
    playlist_length = len(expected)
    s = seed
    for i in range(playlist_length):
        swap_index = s % playlist_length
        if "Rock" in expected[i] or "Rock" in expected[swap_index]:
            s = (s * 16807) % 2147483647
            continue
        expected[i], expected[swap_index] = expected[swap_index], expected[i]
        s = (s * 16807) % 2147483647

    assert candidate(playlist[:], seed) == expected

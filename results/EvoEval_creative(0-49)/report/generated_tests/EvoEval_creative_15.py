# Final accepted test suite for EvoEval_creative/15
# 6 test function(s), mutation score computed over 25 mutant(s)

def check(candidate):
    assert candidate(['5H', '5D', '5S', '9C', '9D']) == 19
    assert candidate(['AS', '2H', '3S', '4H', '5D']) == 0
    assert candidate(['KH', 'KD', 'KS', 'KC', 'KA']) == 52
    assert candidate([]) == 0

def check(candidate):
    hand = ['AH', 'AD', 'AS', '9C', '9D']
    result = candidate(hand)
    # compute expected according to the canonical spec
    card_values = {'A': 1, 'J': 11, 'Q': 12, 'K': 13}
    card_values.update({str(i): i for i in range(2, 11)})
    counts = {}
    for card in hand:
        counts[card[:-1]] = counts.get(card[:-1], 0) + 1
    expected = 0
    for card, count in counts.items():
        if count == 2:
            expected += card_values[card]
        elif count == 3:
            expected += 2 * card_values[card]
        elif count == 4:
            expected += 3 * card_values[card]
        elif count == 5:
            expected += 4 * card_values[card]
    assert result == expected

def check(candidate):
    hand = ['QH', 'QD', '2S', '3H', '4C']
    result = candidate(hand)
    assert result == 12

def check(candidate):
    hand = ['7H', '7D', '7S', '7C', '3D']
    assert candidate(hand) == 21

def check(candidate):
    hand = ['JH', 'JD', '3C', '4S', '5D']
    result = candidate(hand)
    # compute expected per specification
    card_values = {'A': 1, 'J': 11, 'Q': 12, 'K': 13}
    card_values.update({str(i): i for i in range(2, 11)})
    counts = {}
    for card in hand:
        counts[card[:-1]] = counts.get(card[:-1], 0) + 1
    expected = 0
    for val, cnt in counts.items():
        if cnt == 2:
            expected += card_values[val]
        elif cnt == 3:
            expected += 2 * card_values[val]
        elif cnt == 4:
            expected += 3 * card_values[val]
        elif cnt == 5:
            expected += 4 * card_values[val]
    assert result == expected

def check(candidate):
    assert candidate(['2H', '2D', '2S', '2C', '2H']) == 8
    assert candidate(['AH', 'AD', 'AS', '7C', '7D']) == 9
    assert candidate(['10H', '10D', '10S', 'JC', 'JD']) == 31

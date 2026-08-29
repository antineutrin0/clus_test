# Final accepted test suite for EvoEval_creative/43
# 2 test function(s), mutation score computed over 18 mutant(s)

def check(candidate):
    assert candidate(4.22, 0.5, 10, 9.8, 100) == 'Insufficient fuel'
    assert candidate(2.5, 1, 2, 3.7, 1.5) == 0.61
    assert candidate(1.0, 1.0, 1, 1.0, 1) == 0.9
    assert candidate(-1.0, -1.0, -1, -1.0, -1) == 'Insufficient fuel'
    assert candidate(0.5, 0.5, 2, 0.5, 2) == 1.9

def check(candidate):
    # total_fuel_consumption = (distance/speed) * (spaceship_weight * planet_gravity * 0.1)
    # choose values so total_fuel_consumption == fuel: trip_duration=1.0, weight*gravity*0.1=0.1 -> fuel=0.1
    assert candidate(1.0, 1.0, 0.1, 1.0, 1.0) == 0.0

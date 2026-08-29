# Baseline iterative tests for EvoEval_creative/6

def baseline_check_0(candidate):
    # Reveal on a 1-row grid (also exercises x=0, y=0 neighborhood handling)
    assert candidate([[-1, -1, 1]], (0, 0)) == [[0, 0, 1]]

    # Clicking an already revealed empty cell (0) returns the initial grid
    assert candidate([[0, -1], [-1, -1]], (0, 0)) == [[0, -1], [-1, -1]]

    # Clicking a mine returns the initial grid (no reveals around it)
    assert candidate([[-1, 1, -1], [-1, -1, -1], [-1, -1, -1]], (0, 1)) == [
        [-1, 1, -1],
        [-1, -1, -1],
        [-1, -1, -1],
    ]

    # Out-of-grid (negative index) returns the initial grid
    assert candidate([[-1]], (-1, 0)) == [[-1]]

    # Out-of-grid (y == cols) returns the initial grid
    assert candidate([[-1, -1]], (0, 2)) == [[-1, -1]]

    # Reveal a 3x3 neighborhood around an unrevealed empty cell; mines stay as 1
    assert candidate(
        [
            [-1, -1, -1, -1],
            [-1, 1, -1, -1],
            [-1, -1, -1, -1],
            [-1, -1, -1, -1],
        ],
        (2, 2),
    ) == [
        [-1, -1, -1, -1],
        [-1, 1, 0, 0],
        [-1, 0, 0, 0],
        [-1, 0, 0, 0],
    ]

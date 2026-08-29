# Baseline iterative tests for EvoEval_creative/26

def baseline_check_0(candidate):
    # Spec example with known exact path
    maze = [[0, 0, 0], [1, 1, 0], [1, 1, 0]]
    assert candidate(maze, (0, 0), (2, 2)) == ["right", "right", "down", "down"]

    def assert_valid_path(maze, start, end, path):
        assert isinstance(path, list)
        moves = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
        x, y = start
        for step in path:
            assert step in moves
            dx, dy = moves[step]
            x, y = x + dx, y + dy
            assert 0 <= x < len(maze) and 0 <= y < len(maze[0])
            assert maze[x][y] == 0
        assert (x, y) == end

    # Path requires correct "up" semantics (no jumping/diagonals) and triggers bottom-boundary exploration
    maze_up = [[0], [0], [0]]
    res_up = candidate(maze_up, (2, 0), (0, 0))
    assert res_up != []
    assert_valid_path(maze_up, (2, 0), (0, 0), res_up)

    # Path requires correct "left" semantics (no jumping/diagonals) and triggers right-boundary exploration
    maze_left = [[0, 0, 0]]
    res_left = candidate(maze_left, (0, 2), (0, 0))
    assert res_left != []
    assert_valid_path(maze_left, (0, 2), (0, 0), res_left)

    # No-path cases (should be empty), also catch wraparound boundary mutants
    assert candidate([[0], [1], [0]], (0, 0), (2, 0)) == []
    assert candidate([[0, 1, 0]], (0, 0), (0, 2)) == []

def baseline_check_1(candidate):
    maze = [
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 0],
    ]
    res = candidate(maze, (0, 0), (2, 0))
    assert res == []

def baseline_check_2(candidate):
    maze = [
        [0, 0],
        [0, 0],
        [0, 0, 0, 0],
    ]
    assert candidate(maze, (0, 0), (1, 0)) == ["down"]

    maze2 = [
        [0, 0],
        [0, 0],
    ]
    assert candidate(maze2, (0, 0), (1, 1)) == ["right", "down"]

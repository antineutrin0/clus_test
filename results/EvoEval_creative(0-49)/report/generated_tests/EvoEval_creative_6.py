# Final accepted test suite for EvoEval_creative/6
# 6 test function(s), mutation score computed over 24 mutant(s)

def check(candidate):
    assert candidate([[0, 1, -1], [1, -1, 0], [0, 1, 1]], (1, 2)) == [[0, 1, -1], [1, -1, 0], [0, 1, 1]]
    assert candidate([[0, 1, -1], [1, -1, 0], [0, 1, 1]], (2, 2)) == [[0, 1, -1], [1, -1, 0], [0, 1, 1]]
    assert candidate([[0, 1, -1], [1, -1, 0], [0, 1, 1]], (1, 1)) == [[0, 1, 0], [1, 0, 0], [0, 1, 1]]
    try:
        assert candidate((1, 0)) == None
    except Exception as e:
        print(e)

def check(candidate):
    # Distinguish CONSTANT_CHANGE: ensure extra upward row would differ if start index changed
    grid1 = [[-1, -1, -1],
             [-1, -1, -1],
             [-1, -1, -1]]
    grid1_in = [row[:] for row in grid1]
    out1 = candidate(grid1_in, (2, 1))
    assert out1 == [[-1, -1, -1],
                    [0, 0, 0],
                    [0, 0, 0]]

    # Distinguish BOUNDARY_SHIFT: clicking with y == cols should be treated as out-of-bounds
    grid2 = [[-1, 1],
             [1, -1]]
    grid2_in = [row[:] for row in grid2]
    out2 = candidate([row[:] for row in grid2], (0, 2))
    assert out2 == grid2_in

def check(candidate):
    def minesweeper_ref(grid, position):
        x, y = position
        rows, cols = len(grid), len(grid[0])
        grid_ = [row[:] for row in grid]
        if x < 0 or x >= rows or y < 0 or y >= cols or (grid[x][y] == 0):
            return grid_
        if grid[x][y] == 1:
            return grid_
        for i in range(max(0, x - 1), min(rows, x + 2)):
            for j in range(max(0, y - 1), min(cols, y + 2)):
                if grid[i][j] != 1:
                    grid_[i][j] = 0
        return grid_
    grid = [[-1, -1, 1], [-1, -1, -1], [1, -1, -1]]
    pos = (0, 1)
    expected = minesweeper_ref([row[:] for row in grid], pos)
    out = candidate([row[:] for row in grid], pos)
    assert out == expected

def check(candidate):
    def reference(grid, position):
        x, y = position
        rows, cols = (len(grid), len(grid[0]))
        grid_ = grid.copy()
        if x < 0 or x >= rows or y < 0 or (y >= cols) or (grid[x][y] == 0):
            return grid_
        if grid[x][y] == 1:
            return grid_
        for i in range(max(0, x - 1), min(rows, x + 2)):
            for j in range(max(0, y - 1), min(cols, y + 2)):
                if grid[i][j] != 1:
                    grid_[i][j] = 0
        return grid_
    grid = [
        [-1, -1, -1, -1],
        [-1, -1, -1, -1],
        [-1, -1, -1, -1]
    ]
    pos = (1, 2)
    inp_for_candidate = [row[:] for row in grid]
    expected = reference([row[:] for row in grid], pos)
    out = candidate(inp_for_candidate, pos)
    assert out == expected

def check(candidate):
    grid = [[-1, 1], [-1, -1]]
    pos = (0, 0)
    inp = [row[:] for row in grid]
    out = candidate(inp, pos)
    # build expected according to canonical specification
    x, y = pos
    rows, cols = len(grid), len(grid[0])
    if x < 0 or x >= rows or y < 0 or y >= cols or grid[x][y] == 0 or grid[x][y] == 1:
        expected = [row[:] for row in grid]
    else:
        expected = [row[:] for row in grid]
        for i in range(max(0, x - 1), min(rows, x + 2)):
            for j in range(max(0, y - 1), min(cols, y + 2)):
                if grid[i][j] != 1:
                    expected[i][j] = 0
    assert out == expected

def check(candidate):
    grid = [[-1, -1, -1],
            [-1, -1, -1],
            [-1, -1, -1]]
    out = candidate([row[:] for row in grid], (-1, 1))
    assert out == grid

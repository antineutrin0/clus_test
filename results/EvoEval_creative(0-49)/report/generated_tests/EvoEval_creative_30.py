# Final accepted test suite for EvoEval_creative/30
# 4 test function(s), mutation score computed over 24 mutant(s)

def check(candidate):
    def manhattan(a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])
    # Target INDEX_BOUNDARY: single-row maze (mutant would access maze[1] -> IndexError)
    maze1 = [[1,1,1,1]]
    start1 = (0,0)
    end1 = (0,3)
    path1 = candidate(maze1, start1, end1)
    assert isinstance(path1, list)
    assert path1 and path1[0] == start1 and path1[-1] == end1
    for (x,y) in path1:
        assert 0 <= x < len(maze1) and 0 <= y < len(maze1[0])
        assert maze1[x][y] == 1
    for a,b in zip(path1, path1[1:]):
        assert abs(a[0]-b[0]) + abs(a[1]-b[1]) == 1
    assert len(path1)-1 == manhattan(start1, end1)

    # Target CONSTANT_CHANGE: require a straight-down shortest path
    maze2 = [[1,1],
             [1,1],
             [1,1]]
    start2 = (0,0)
    end2 = (2,0)
    path2 = candidate(maze2, start2, end2)
    assert isinstance(path2, list)
    assert path2 and path2[0] == start2 and path2[-1] == end2
    for (x,y) in path2:
        assert 0 <= x < len(maze2) and 0 <= y < len(maze2[0])
        assert maze2[x][y] == 1
    for a,b in zip(path2, path2[1:]):
        assert abs(a[0]-b[0]) + abs(a[1]-b[1]) == 1
    assert len(path2)-1 == manhattan(start2, end2)

def check(candidate):
    maze = [[1, 0], [0, 1]]
    start = (1, 1)
    end = (0, 0)
    path = candidate(maze, start, end)
    assert isinstance(path, list)
    assert path == []

def check(candidate):
    maze = [[1], [1]]
    start = (1, 0)
    end = (0, 0)
    path = candidate(maze, start, end)
    from collections import deque
    rows, cols = len(maze), len(maze[0])
    visited = [[False] * cols for _ in range(rows)]
    queue = deque([(start, [start])])
    expected = []
    while queue:
        (x, y), p = queue.popleft()
        if (x, y) == end:
            expected = p
            break
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = (x + dx, y + dy)
            if 0 <= nx < rows and 0 <= ny < cols and maze[nx][ny] == 1 and (not visited[nx][ny]):
                queue.append(((nx, ny), p + [(nx, ny)]))
                visited[nx][ny] = True
    assert path == expected

def check(candidate):
    maze = [
        [0,0,0],
        [0,0,0],
        [1,1,1]
    ]
    start = (2, 2)
    end = (2, 0)
    path = candidate(maze, start, end)
    assert path != []

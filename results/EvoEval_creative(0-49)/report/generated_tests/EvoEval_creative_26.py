# Final accepted test suite for EvoEval_creative/26
# 7 test function(s), mutation score computed over 24 mutant(s)

def check(candidate):
    assert candidate([[0, 0, 0], [1, 1, 0], [1, 1, 0]], (0, 0), (2, 2)) == ['right', 'right', 'down', 'down']
    assert candidate([[0, 1, 0], [0, 1, 0], [0, 1, 0]], (0, 0), (0, 2)) == []
    assert candidate([[0, 0, 0]], (0, 0), (0, 0)) == []
    assert candidate([[0, 0, 0], [0, 1, 0], [0, 0, 0]], (0, 0), (2, 0)) == ['down', 'down']

    def pathfinder(maze, s, e):
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        dirnames = ['right', 'left', 'down', 'up']
        q = [(s, [])]
        vis = set()
        while q:
            (x, y), p = q.pop(0)
            if (x, y) == e:
                return p
            if (x, y) in vis:
                continue
            vis.add((x, y))
            for d, dname in zip(directions, dirnames):
                dx, dy = d
                nx, ny = (x + dx, y + dy)
                if 0 <= nx < len(maze) and 0 <= ny < len(maze[0]) and (maze[nx][ny] == 0):
                    q.append(((nx, ny), p + [dname]))
    assert pathfinder([[0, 0, 0], [1, 1, 0], [1, 0, 0]], (0, 0), (2, 2)) == ['right', 'right', 'down', 'down'], 'Pathfinding works correctly for example case 1'
    assert pathfinder([[1, 1, 1], [1, 1, 1]], (0, 0), (0, 0)) == [], 'No path exists for blocked maze'
    assert pathfinder([[0, 0, 0], [0, 1, 0], [0, 0, 0]], (0, 0), (2, 0)) == ['down', 'down'], 'Correct path for example case 2'

def check(candidate):
    assert candidate([[0, 0]], (0, 1), (0, 0)) == ['left']

def check(candidate):
    assert candidate([[1, 0], [0, 1]], (1, 0), (0, 1)) == []

def check(candidate):
    maze = [
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 0],
    ]
    assert candidate(maze, (0, 0), (2, 2)) == []

def check(candidate):
    def ref_pathfinder(maze, start, end):
        if start == end:
            return []
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        names = ['right', 'left', 'down', 'up']
        queue = [(start, [])]
        visited = set()
        rows, cols = len(maze), len(maze[0])
        while queue:
            (x, y), path = queue.pop(0)
            if (x, y) == end:
                return path
            if (x, y) in visited:
                continue
            visited.add((x, y))
            for (dx, dy), name in zip(directions, names):
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols and maze[nx][ny] == 0:
                    queue.append(((nx, ny), path + [name]))
        return []
    maze = [[0], [0], [0]]
    start = (2, 0)
    end = (0, 0)
    expected = ref_pathfinder(maze, start, end)
    assert candidate(maze, start, end) == expected

def check(candidate):
    maze = [
        [0,0,1,1,1,1],
        [0,0,1,0,0,0],
        [1,0,1,0,1,0],
        [1,0,0,0,1,0],
        [1,1,1,0,1,0],
        [1,1,1,0,0,0],
    ]
    start = (0, 0)
    end = (5, 5)
    # compute true shortest path length with BFS (independent of candidate)
    from collections import deque
    dirs = {'right': (0,1), 'left': (0,-1), 'down': (1,0), 'up': (-1,0)}
    q = deque([(start, 0)])
    seen = {start}
    shortest = None
    rows, cols = len(maze), len(maze[0])
    while q:
        (x,y), d = q.popleft()
        if (x,y) == end:
            shortest = d
            break
        for dx,dy in dirs.values():
            nx, ny = x+dx, y+dy
            if 0 <= nx < rows and 0 <= ny < cols and maze[nx][ny] == 0 and (nx,ny) not in seen:
                seen.add((nx,ny))
                q.append(((nx,ny), d+1))
    assert shortest is not None, "sanity: test maze must have a path"
    result = candidate(maze, start, end)
    assert isinstance(result, list)
    assert len(result) == shortest

def check(candidate):
    maze = [
        [0, 0, 0],
        [1, 1],
    ]
    start = (0, 0)
    end = (0, 2)

    path = candidate(maze, start, end)
    assert path != []

    r, c = start
    moves = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
    for step in path:
        assert step in moves
        dr, dc = moves[step]
        r, c = r + dr, c + dc
        assert 0 <= r < len(maze)
        assert 0 <= c < len(maze[r])
        assert maze[r][c] == 0
    assert (r, c) == end

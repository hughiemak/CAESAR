import numpy as np
import copy

from typing import List, Optional

def is_valid(board: List[List[str]], max_size: int) -> bool:
    frontier, discovered = [], set()
    frontier.append((0, 0))
    while frontier:
        r, c = frontier.pop()
        if not (r, c) in discovered:
            discovered.add((r, c))
            directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
            for x, y in directions:
                r_new = r + x
                c_new = c + y
                if r_new < 0 or r_new >= max_size or c_new < 0 or c_new >= max_size:
                    continue
                if board[r_new][c_new] == "G":
                    return True
                if board[r_new][c_new] != "H":
                    frontier.append((r_new, c_new))
    return False


def generate_random_map(size: int = 8, n_holes: int = 4) -> List[str]:
    """Generates a random valid map (one that has a path from start to goal)
    Args:
        size: size of each side of the grid
        p: probability that a tile is frozen
    Returns:
        A random valid map
    """
    valid = False
    board = []  # initialize to make pyright happy
    
    def count_holes(b):
        count = 0
        for line in b:
            for c in line:
                if c == 'H':
                    count += 1
        return count

    while not valid:
        p = 0.5
        board = np.random.choice(["F", "H"], (size, size), p=[p, 1 - p])
        board[0][0] = "S"
        board[-1][-1] = "G"
        valid = is_valid(board, size) and count_holes(board)==n_holes
    return ["".join(x) for x in board]

def are_identical_boards(b1, b2):
    identical = True
    n_rows, n_cols = len(b1), len(b1[0])
    for i in range(n_rows):
        for j in range(n_cols):
            if b1[i][j] != b2[i][j]: identical = False
    return identical

def generate_two_random_maps(size):
    identical = True
    while identical:
        m1 = generate_random_map(size=size)
        m2 = generate_random_map(size=size)
        identical = are_identical_boards(m1, m2)
    return m1, m2

def generate_two_opposing_maps(size):
    assert size == 4, f"Opposing maps with size {size}  not supported."
    
    m1 = ["SFGH", "FHHH", "FFFF", "FFFF"]
    m2 = ["SFFH", "FHHH", "FFFF", "FFFG"]

    return m1, m2

def generate_two_identical_maps(size):
    m1 = generate_random_map(size=size)
    m2 = copy.deepcopy(m1)
    return m1, m2
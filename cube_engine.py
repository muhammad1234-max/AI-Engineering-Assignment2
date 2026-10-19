"""Root-level export proxy for cube_engine.

Enables direct imports via `import cube_engine` or `from cube_engine import Cube`.
"""

from backend.app.cube_engine import (
    ALL_MOVES,
    FACES,
    FACE_NAMES,
    MOVE_DESCRIPTIONS,
    MOVE_PERMUTATIONS,
    OPPOSITE_FACES,
    SOLVED_STATE,
    BFSSolveResult,
    Cube,
    compute_optimality_gap,
    generate_scramble,
    invert_move,
    invert_moves,
    solve_bfs,
)

__all__ = [
    "ALL_MOVES",
    "FACES",
    "FACE_NAMES",
    "MOVE_DESCRIPTIONS",
    "MOVE_PERMUTATIONS",
    "OPPOSITE_FACES",
    "SOLVED_STATE",
    "BFSSolveResult",
    "Cube",
    "compute_optimality_gap",
    "generate_scramble",
    "invert_move",
    "invert_moves",
    "solve_bfs",
]

if __name__ == "__main__":
    print("=" * 60)
    print("Deterministic 3x3 Rubik's Cube Engine Demo")
    print("=" * 60)

    cube = Cube()
    print("Initial Solved State:", cube.is_solved())
    print("Facelet string (54 chars):", cube.to_facelet_string())
    print("\nUnfolded 2D Net:")
    print(cube.get_2d_net())

    # Demo a scramble
    scramble = cube.scramble(length=4, seed=42)
    print(f"\nApplied 4-move scramble: {' '.join(scramble)}")
    print("Is solved now?", cube.is_solved())
    print(cube.get_2d_net())

    # Solve with exact BFS
    result = cube.solve_shortest_path()
    print(f"\nExact BFS Shortest-Path Result:")
    print(f"  Solved: {result.is_solved}")
    print(f"  Shortest distance: {result.distance}")
    print(f"  Solution moves: {' '.join(result.moves)}")
    print(f"  Nodes explored: {result.nodes_explored}")
    print(f"  Execution time: {result.execution_time_ms:.2f} ms")

    # Apply solution and verify
    cube.apply_moves(result.moves)
    print("\nAfter applying BFS solution moves:")
    print("Is solved:", cube.is_solved())
    print("=" * 60)

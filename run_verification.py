"""Standalone verification script for Phase 1 Cube Engine and Exact BFS Solver.

Demonstrates and verifies:
1. State Representation and 54-facelet serialization.
2. Action Space & move execution for all 18 Singmaster face turns.
3. Random Scramble Generator avoiding redundant consecutive face turns.
4. Exact BFS Shortest-Path calculation and Optimality Gap.
"""

import time
from backend.app.cube_engine import (
    ALL_MOVES,
    MOVE_DESCRIPTIONS,
    SOLVED_STATE,
    Cube,
    compute_optimality_gap,
    generate_scramble,
    solve_bfs,
)


def header(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def verify_phase_1():
    header("PHASE 1 VERIFICATION: DETERMINISTIC CUBE ENGINE & EXACT BFS SOLVER")

    # 1. State Representation
    print("\n[1] STATE REPRESENTATION & SERIALIZATION:")
    cube = Cube()
    print(f"  - Initial is_solved(): {cube.is_solved()}")
    state_str = cube.to_facelet_string()
    print(f"  - Serialized string (54 chars): {state_str}")
    assert len(state_str) == 54, "State string must be exactly 54 characters"
    assert state_str == SOLVED_STATE, "State must match standard solved string"
    print("  - 2D Unfolded Net View:")
    for line in cube.get_2d_net().splitlines():
        print(f"      {line}")

    # 2. Action Space
    print("\n[2] ACTION SPACE & MOVE VERIFICATION:")
    print(f"  - Total standard Singmaster moves: {len(ALL_MOVES)}")
    print("  - Move samples with human-readable descriptions:")
    for sample in ["U", "U'", "U2", "R", "F'", "D2"]:
        print(f"      * {sample:<3} : {MOVE_DESCRIPTIONS[sample]}")

    # Verify periodicity M^4 = I and commutator
    c_period = Cube()
    for _ in range(4):
        c_period.apply_move("R")
    assert c_period.is_solved(), "R^4 must return to solved state"
    print("  - Move periodicity check (R^4 == Solved): PASSED")

    # 3. Scramble Generator
    print("\n[3] SCRAMBLE GENERATION:")
    for label, length in [
        ("Warm-up (3 moves)", 3),
        ("Primary Benchmark (4 moves)", 4),
        ("Primary Benchmark (5 moves)", 5),
        ("Extension (15 moves)", 15),
    ]:
        scramble = generate_scramble(length=length, seed=42 + length)
        print(f"  - {label:<28}: {' '.join(scramble)}")
        # Check no consecutive same-face turns
        for i in range(len(scramble) - 1):
            assert scramble[i][0] != scramble[i + 1][0], "Consecutive same-face turn detected!"
    print("  - Redundant consecutive move check: ALL CLEAN & PASSED")

    # 4. Exact BFS Shortest-Path Solver
    print("\n[4] EXACT BFS SHORTEST-PATH SOLVER:")
    for length in [1, 2, 3, 4, 5]:
        scramble = generate_scramble(length=length, seed=100 + length)
        test_cube = Cube()
        test_cube.apply_moves(scramble)

        t0 = time.perf_counter()
        result = test_cube.solve_shortest_path(max_depth=7)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Apply the computed moves to verify they actually solve the cube
        verification_cube = test_cube.clone()
        verification_cube.apply_moves(result.moves)
        assert verification_cube.is_solved(), f"BFS solution failed to solve the cube!"
        assert result.distance <= length, "BFS distance must be <= scramble length"

        print(
            f"  - Length {length} scramble: {' '.join(scramble):<18} "
            f"-> Distance: {result.distance} | Solution: {' '.join(result.moves):<16} "
            f"| Time: {elapsed_ms:>5.2f}ms | Nodes: {result.nodes_explored}"
        )

    # 5. Optimality Gap Calculation Demo
    print("\n[5] OPTIMALITY GAP CALCULATION:")
    scramble_demo = ["R", "U", "R'"]
    demo_cube = Cube()
    demo_cube.apply_moves(scramble_demo)
    bfs_res = demo_cube.solve_shortest_path()

    # Suppose Model A found the optimal 3 moves: ["R", "U'", "R'"]
    model_a_moves = ["R", "U'", "R'"]
    # Suppose Model B took 5 moves (e.g., redundant moves)
    model_b_moves = ["R", "U'", "U", "U'", "R'"]

    gap_a = compute_optimality_gap(model_a_moves, bfs_res.distance)
    gap_b = compute_optimality_gap(model_b_moves, bfs_res.distance)

    print(f"  - Scramble: {' '.join(scramble_demo)}")
    print(f"  - Exact BFS Shortest Distance: {bfs_res.distance}")
    print(f"  - Model A (optimal): moves={model_a_moves} -> Optimality Gap = {gap_a}")
    print(f"  - Model B (suboptimal): moves={model_b_moves} -> Optimality Gap = {gap_b}")

    header("ALL PHASE 1 CHECKS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    verify_phase_1()

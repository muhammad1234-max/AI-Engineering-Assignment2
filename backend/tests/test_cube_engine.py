"""Comprehensive Unit and Integration Tests for Cube Engine and BFS Solver."""

import pytest
from backend.app.cube_engine import (
    ALL_MOVES,
    FACES,
    MOVE_DESCRIPTIONS,
    SOLVED_STATE,
    BFSSolveResult,
    Cube,
    compute_optimality_gap,
    generate_scramble,
    invert_move,
    invert_moves,
    solve_bfs,
)


class TestStateRepresentation:
    """Tests for 54-facelet state representation and serialization."""

    def test_initial_cube_is_solved(self):
        cube = Cube()
        assert cube.is_solved() is True
        assert cube.to_facelet_string() == SOLVED_STATE

    def test_facelet_string_structure(self):
        cube = Cube()
        s = cube.to_facelet_string()
        assert len(s) == 54
        for face in FACES:
            assert s.count(face) == 9

    def test_from_facelet_string_roundtrip(self):
        cube1 = Cube()
        cube1.apply_move("R")
        s = cube1.to_facelet_string()

        cube2 = Cube.from_facelet_string(s)
        assert cube2.to_facelet_string() == s
        assert cube2.is_solved() is False

    def test_invalid_state_string_validation(self):
        # Invalid length
        with pytest.raises(ValueError, match="exactly 54 characters"):
            Cube("UUU")

        # Wrong character counts
        with pytest.raises(ValueError, match="appears"):
            Cube("U" * 54)

        # Invalid type
        with pytest.raises(TypeError, match="must be a string"):
            Cube(12345)  # type: ignore

    def test_clone_creates_independent_copy(self):
        original = Cube()
        clone = original.clone()
        clone.apply_move("U")

        assert original.is_solved() is True
        assert clone.is_solved() is False
        assert original.to_facelet_string() != clone.to_facelet_string()

    def test_reset_restores_solved_state(self):
        cube = Cube()
        cube.apply_moves("R U R' U'")
        assert cube.is_solved() is False
        cube.reset()
        assert cube.is_solved() is True
        assert cube.to_facelet_string() == SOLVED_STATE

    def test_2d_net_rendering(self):
        cube = Cube()
        net = cube.get_2d_net()
        assert "+---+" in net
        assert "|UUU|" in net
        assert "|FFF|" in net
        assert "|DDD|" in net


class TestActionSpace:
    """Tests for the 18 standard Singmaster face turns."""

    def test_all_18_moves_defined(self):
        assert len(ALL_MOVES) == 18
        assert len(MOVE_DESCRIPTIONS) == 18
        for m in ALL_MOVES:
            assert m in MOVE_DESCRIPTIONS
            desc = MOVE_DESCRIPTIONS[m]
            assert isinstance(desc, str)
            assert len(desc) > 10

    def test_invalid_move_raises_error(self):
        cube = Cube()
        with pytest.raises(ValueError, match="Invalid move"):
            cube.apply_move("INVALID")
        with pytest.raises(ValueError, match="Invalid move"):
            cube.apply_move("M")  # Middle slice moves not in standard 18 face turns

    def test_move_order_four(self):
        """Any 90-degree turn applied 4 times must return to the initial state."""
        for face in FACES:
            cube = Cube()
            for _ in range(4):
                cube.apply_move(face)
            assert cube.is_solved() is True, f"{face}^4 failed to restore solved state"

    def test_inverse_moves(self):
        """Applying M followed by M' must return to the initial state."""
        for m in ["U", "D", "L", "R", "F", "B"]:
            cube = Cube()
            cube.apply_move(m)
            cube.apply_move(invert_move(m))
            assert cube.is_solved() is True, f"{m} followed by {invert_move(m)} failed"

    def test_half_turns(self):
        """M followed by M must equal M2, and (M2)^2 must return to solved."""
        for face in FACES:
            cube1 = Cube()
            cube1.apply_move(face).apply_move(face)

            cube2 = Cube()
            cube2.apply_move(face + "2")

            assert cube1.to_facelet_string() == cube2.to_facelet_string()

            cube2.apply_move(face + "2")
            assert cube2.is_solved() is True

    def test_sexy_move_commutator(self):
        """The 'sexy move' (R U R' U') has order 6."""
        cube = Cube()
        for i in range(1, 7):
            cube.apply_moves("R U R' U'")
            if i < 6:
                assert cube.is_solved() is False
        assert cube.is_solved() is True


class TestScrambleGenerator:
    """Tests for valid non-redundant scramble generation."""

    def test_scramble_lengths(self):
        for length in [0, 1, 3, 4, 5, 10, 20]:
            scramble = generate_scramble(length=length)
            assert len(scramble) == length

    def test_no_consecutive_same_face_turns(self):
        """Scrambles must never contain consecutive turns on the same face."""
        for seed in range(50):
            scramble = generate_scramble(length=20, seed=seed)
            for i in range(len(scramble) - 1):
                face1 = scramble[i][0]
                face2 = scramble[i + 1][0]
                assert face1 != face2, (
                    f"Consecutive same face found in seed {seed}: {scramble[i]} followed by {scramble[i+1]}"
                )

    def test_no_commuting_face_redundancy(self):
        """Scrambles must avoid patterns like U D U."""
        for seed in range(50):
            scramble = generate_scramble(length=20, seed=seed, avoid_opposite_redundancy=True)
            for i in range(len(scramble) - 2):
                f1, f2, f3 = scramble[i][0], scramble[i + 1][0], scramble[i + 2][0]
                if f1 == f3:
                    # If f1 == f3, f2 cannot be the opposite face of f1
                    from backend.app.cube_engine import OPPOSITE_FACES
                    assert OPPOSITE_FACES.get(f1) != f2, (
                        f"Redundant commuting sequence {f1}-{f2}-{f3} in scramble {' '.join(scramble)}"
                    )

    def test_deterministic_seed(self):
        s1 = generate_scramble(length=10, seed=12345)
        s2 = generate_scramble(length=10, seed=12345)
        assert s1 == s2

        s3 = generate_scramble(length=10, seed=54321)
        assert s1 != s3


class TestExactBFSSolver:
    """Tests for exact Breadth-First Search shortest path solver."""

    def test_solve_already_solved_cube(self):
        cube = Cube()
        result = cube.solve_shortest_path()
        assert result.is_solved is True
        assert result.distance == 0
        assert result.moves == []

    def test_solve_depth_1_scrambles(self):
        for move in ALL_MOVES:
            cube = Cube()
            cube.apply_move(move)
            result = cube.solve_shortest_path(max_depth=3)
            assert result.is_solved is True
            assert result.distance == 1
            assert len(result.moves) == 1
            # Inverse of move
            assert result.moves[0] == invert_move(move)

    def test_solve_depth_2_scrambles(self):
        scramble = ["R", "U"]
        cube = Cube()
        cube.apply_moves(scramble)
        result = cube.solve_shortest_path(max_depth=4)
        assert result.is_solved is True
        assert result.distance == 2
        # Verify moves actually solve it
        cube.apply_moves(result.moves)
        assert cube.is_solved() is True

    def test_solve_depth_3_scrambles(self):
        scramble = ["F", "R2", "D'"]
        cube = Cube()
        cube.apply_moves(scramble)
        result = cube.solve_shortest_path(max_depth=5)
        assert result.is_solved is True
        assert result.distance == 3
        cube.apply_moves(result.moves)
        assert cube.is_solved() is True

    def test_solve_depth_4_benchmark_scrambles(self):
        scramble = generate_scramble(length=4, seed=42)
        cube = Cube()
        cube.apply_moves(scramble)
        result = cube.solve_shortest_path(max_depth=6)
        assert result.is_solved is True
        assert result.distance == 4
        cube.apply_moves(result.moves)
        assert cube.is_solved() is True

    def test_solve_depth_5_benchmark_scrambles(self):
        scramble = generate_scramble(length=5, seed=77)
        cube = Cube()
        cube.apply_moves(scramble)
        result = cube.solve_shortest_path(max_depth=7)
        assert result.is_solved is True
        assert result.distance <= 5  # Distance could be 5 or less if cancellation
        cube.apply_moves(result.moves)
        assert cube.is_solved() is True

    def test_unidirectional_vs_bidirectional_equivalence(self):
        """Both algorithms must yield the exact same shortest distance."""
        scramble = ["U", "R'", "F2"]
        cube = Cube()
        cube.apply_moves(scramble)

        res_uni = solve_bfs(cube, max_depth=4, bidirectional=False)
        res_bi = solve_bfs(cube, max_depth=4, bidirectional=True)

        assert res_uni.is_solved is True
        assert res_bi.is_solved is True
        assert res_uni.distance == res_bi.distance == 3

    def test_optimality_gap_calculation(self):
        # Optimal case
        assert compute_optimality_gap(model_moves=["U'", "R'"], shortest_distance=2) == 0
        # Suboptimal case: took 4 moves when shortest was 2
        assert compute_optimality_gap(model_moves=["U", "U2", "R", "R2"], shortest_distance=2) == 2

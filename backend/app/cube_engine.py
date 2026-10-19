"""Deterministic 3x3 Rubik's Cube Engine and Exact BFS Solver.

This module implements:
1. Deterministic State Representation (54-facelet string: U, R, F, D, L, B).
2. Action Space (18 standard Singmaster face turns with human-readable descriptions).
3. Non-redundant Random Scramble Generator.
4. Exact Shortest-Path Solver using Breadth-First Search (BFS) / Meet-in-the-Middle BiBFS.
5. Optimality Gap computation for benchmark evaluations.
"""

from __future__ import annotations

import collections
import random
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple, Union

# ---------------------------------------------------------------------------
# Constants & Face Definitions
# ---------------------------------------------------------------------------

FACES = ("U", "R", "F", "D", "L", "B")
FACE_NAMES = {
    "U": "Up (White)",
    "R": "Right (Red)",
    "F": "Front (Green)",
    "D": "Down (Yellow)",
    "L": "Left (Orange)",
    "B": "Back (Blue)",
}

# Standard solved state: 54 characters, 9 for each face in standard URFDLB order.
SOLVED_STATE = "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB"

# Singmaster move descriptions
MOVE_DESCRIPTIONS: Dict[str, str] = {
    "U": "Turn Up face clockwise 90 degrees",
    "U'": "Turn Up face counter-clockwise 90 degrees",
    "U2": "Turn Up face 180 degrees",
    "D": "Turn Down face clockwise 90 degrees",
    "D'": "Turn Down face counter-clockwise 90 degrees",
    "D2": "Turn Down face 180 degrees",
    "L": "Turn Left face clockwise 90 degrees",
    "L'": "Turn Left face counter-clockwise 90 degrees",
    "L2": "Turn Left face 180 degrees",
    "R": "Turn Right face clockwise 90 degrees",
    "R'": "Turn Right face counter-clockwise 90 degrees",
    "R2": "Turn Right face 180 degrees",
    "F": "Turn Front face clockwise 90 degrees",
    "F'": "Turn Front face counter-clockwise 90 degrees",
    "F2": "Turn Front face 180 degrees",
    "B": "Turn Back face clockwise 90 degrees",
    "B'": "Turn Back face counter-clockwise 90 degrees",
    "B2": "Turn Back face 180 degrees",
}

ALL_MOVES: Tuple[str, ...] = tuple(MOVE_DESCRIPTIONS.keys())

# Opposite face mapping (faces that lie on the same spatial axis and commute)
OPPOSITE_FACES: Dict[str, str] = {
    "U": "D",
    "D": "U",
    "L": "R",
    "R": "L",
    "F": "B",
    "B": "F",
}


# ---------------------------------------------------------------------------
# 3D Coordinate System & Exact Permutation Generation
# ---------------------------------------------------------------------------

def _build_facelet_definitions() -> List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]:
    """Build exact 3D coordinates and normal vectors for all 54 facelets.
    
    Coordinate axes:
        +X: Right, -X: Left
        +Y: Up,    -Y: Down
        +Z: Front, -Z: Back
        
    Face ordering: U (0..8), R (9..17), F (18..26), D (27..35), L (36..44), B (45..53).
    Each face is arranged row-major (top-to-bottom, left-to-right from the perspective
    of someone looking directly at that face).
    """
    facelets = []
    
    # U face (y = +1, normal = (0, 1, 0))
    # Looking down from top: Back is top (-Z), Front is bottom (+Z), Left is -X, Right is +X.
    for z in (-1, 0, 1):
        for x in (-1, 0, 1):
            facelets.append(((x, 1, z), (0, 1, 0)))

    # R face (x = +1, normal = (1, 0, 0))
    # Looking at Right face: Up is +Y, Down is -Y, Front is left (+Z), Back is right (-Z).
    for y in (1, 0, -1):
        for z in (1, 0, -1):
            facelets.append(((1, y, z), (1, 0, 0)))

    # F face (z = +1, normal = (0, 0, 1))
    # Looking at Front face: Up is +Y, Down is -Y, Left is -X, Right is +X.
    for y in (1, 0, -1):
        for x in (-1, 0, 1):
            facelets.append(((x, y, 1), (0, 0, 1)))

    # D face (y = -1, normal = (0, -1, 0))
    # Looking at Down face from below: Front is top (+Z), Back is bottom (-Z), Left is -X, Right is +X.
    for z in (1, 0, -1):
        for x in (-1, 0, 1):
            facelets.append(((x, -1, z), (0, -1, 0)))

    # L face (x = -1, normal = (-1, 0, 0))
    # Looking at Left face: Up is +Y, Down is -Y, Back is left (-Z), Front is right (+Z).
    for y in (1, 0, -1):
        for z in (-1, 0, 1):
            facelets.append(((-1, y, z), (-1, 0, 0)))

    # B face (z = -1, normal = (0, 0, -1))
    # Looking at Back face: Up is +Y, Down is -Y, Right is left (+X), Left is right (-X).
    for y in (1, 0, -1):
        for x in (1, 0, -1):
            facelets.append(((x, y, -1), (0, 0, -1)))

    return facelets


def _build_move_permutations() -> Dict[str, List[int]]:
    """Derive exact 54-facelet permutations for all 18 Singmaster moves.
    
    Each permutation `perm` is a list of length 54 where `new_state[i] = old_state[perm[i]]`.
    """
    facelets = _build_facelet_definitions()
    lookup = {f: i for i, f in enumerate(facelets)}

    # 3D rotation transformations for clockwise 90 degree face turns:
    # U: layer y = 1. Looking down +Y: x' = -z, z' = x, y' = y
    def rot_u(p: Tuple[int, int, int]) -> Tuple[int, int, int]:
        return (-p[2], p[1], p[0])

    # D: layer y = -1. Looking up -Y: x' = z, z' = -x, y' = y
    def rot_d(p: Tuple[int, int, int]) -> Tuple[int, int, int]:
        return (p[2], p[1], -p[0])

    # R: layer x = 1. Looking down +X: y' = z, z' = -y, x' = x
    def rot_r(p: Tuple[int, int, int]) -> Tuple[int, int, int]:
        return (p[0], p[2], -p[1])

    # L: layer x = -1. Looking up -X: y' = -z, z' = y, x' = x
    def rot_l(p: Tuple[int, int, int]) -> Tuple[int, int, int]:
        return (p[0], -p[2], p[1])

    # F: layer z = 1. Looking down +Z: x' = y, y' = -x, z' = z
    def rot_f(p: Tuple[int, int, int]) -> Tuple[int, int, int]:
        return (p[1], -p[0], p[2])

    # B: layer z = -1. Looking up -Z: x' = -y, y' = x, z' = z
    def rot_b(p: Tuple[int, int, int]) -> Tuple[int, int, int]:
        return (-p[1], p[0], p[2])

    layer_check = {
        "U": (lambda p: p[1] == 1, rot_u),
        "D": (lambda p: p[1] == -1, rot_d),
        "R": (lambda p: p[0] == 1, rot_r),
        "L": (lambda p: p[0] == -1, rot_l),
        "F": (lambda p: p[2] == 1, rot_f),
        "B": (lambda p: p[2] == -1, rot_b),
    }

    base_perms: Dict[str, List[int]] = {}
    for face, (in_layer, rot_func) in layer_check.items():
        perm = list(range(54))
        for src_idx, (pos, norm) in enumerate(facelets):
            if in_layer(pos):
                new_pos = rot_func(pos)
                new_norm = rot_func(norm)
                dst_idx = lookup[(new_pos, new_norm)]
                perm[dst_idx] = src_idx
        base_perms[face] = perm

    # Expand to all 18 moves (Clockwise, 180-degree, Counter-clockwise)
    perms: Dict[str, List[int]] = {}
    for face in FACES:
        p1 = base_perms[face]
        perms[face] = p1
        # 180 turn: apply 90 turn twice
        p2 = [p1[p1[i]] for i in range(54)]
        perms[face + "2"] = p2
        # Counter-clockwise 90 turn: apply 90 turn 3 times
        p3 = [p1[p2[i]] for i in range(54)]
        perms[face + "'"] = p3

    return perms


# Precomputed permutation tables for fast O(1) state transitions
MOVE_PERMUTATIONS: Dict[str, List[int]] = _build_move_permutations()


# ---------------------------------------------------------------------------
# Helper Utilities
# ---------------------------------------------------------------------------

def invert_move(move: str) -> str:
    """Return the inverse of a Singmaster move."""
    if move.endswith("'"):
        return move[:-1]
    elif move.endswith("2"):
        return move
    else:
        return move + "'"


def invert_moves(moves: Iterable[str]) -> List[str]:
    """Return the inverted sequence of moves in reverse order."""
    return [invert_move(m) for m in reversed(list(moves))]


def compute_optimality_gap(model_moves: List[str], shortest_distance: int) -> int:
    """Compute the optimality gap: (Model moves count - Exact shortest moves).
    
    A gap of 0 means the model found an optimal shortest path.
    A positive gap indicates suboptimal turns.
    """
    return len(model_moves) - shortest_distance


# ---------------------------------------------------------------------------
# Core Cube Class
# ---------------------------------------------------------------------------

class Cube:
    """A deterministic 3x3 Rubik's Cube state representation and manipulator."""

    __slots__ = ("_facelets",)

    def __init__(self, state: Optional[str] = None) -> None:
        """Initialize the Cube.
        
        Args:
            state: An optional 54-character facelet string. Defaults to solved state.
        """
        if state is None:
            self._facelets = list(SOLVED_STATE)
        else:
            self._validate_state_string(state)
            self._facelets = list(state)

    @staticmethod
    def _validate_state_string(state: str) -> None:
        """Validate that the string is a valid 54-character facelet string."""
        if not isinstance(state, str):
            raise TypeError(f"State must be a string, got {type(state).__name__}")
        if len(state) != 54:
            raise ValueError(f"State must be exactly 54 characters, got {len(state)}")
        counts = collections.Counter(state)
        for face in FACES:
            if counts[face] != 9:
                raise ValueError(
                    f"Invalid facelet string: face '{face}' appears {counts[face]} times (expected 9)"
                )

    def to_facelet_string(self) -> str:
        """Serialize current state into standard 54-character string."""
        return "".join(self._facelets)

    @classmethod
    def from_facelet_string(cls, state: str) -> Cube:
        """Create a new Cube instance from a 54-character facelet string."""
        return cls(state)

    def is_solved(self) -> bool:
        """Check if all 6 faces are solved (each face has a uniform color/character)."""
        # Fast check: compare directly with SOLVED_STATE if center orientation is fixed
        if "".join(self._facelets) == SOLVED_STATE:
            return True
        # General check: each 9-facelet block has uniform character matching its center
        for i in range(6):
            face_slice = self._facelets[i * 9 : (i + 1) * 9]
            if len(set(face_slice)) != 1:
                return False
        return True

    def reset(self) -> None:
        """Reset the cube to the standard solved state."""
        self._facelets = list(SOLVED_STATE)

    def clone(self) -> Cube:
        """Create a deep copy of the current cube state."""
        c = Cube.__new__(Cube)
        c._facelets = list(self._facelets)
        return c

    def apply_move(self, move: str) -> Cube:
        """Apply a single Singmaster move to the cube in-place.
        
        Args:
            move: One of the 18 Singmaster moves (e.g., 'U', "U'", 'U2', etc.)
            
        Returns:
            self (for method chaining)
        """
        if move not in MOVE_PERMUTATIONS:
            raise ValueError(
                f"Invalid move '{move}'. Supported moves are: {', '.join(ALL_MOVES)}"
            )
        perm = MOVE_PERMUTATIONS[move]
        old = self._facelets
        self._facelets = [old[perm[i]] for i in range(54)]
        return self

    def apply_moves(self, moves: Union[str, Iterable[str]]) -> Cube:
        """Apply a sequence of moves to the cube in-place.
        
        Args:
            moves: Either a space-delimited string (e.g. "R U R' U'") or an iterable of moves.
            
        Returns:
            self (for method chaining)
        """
        if isinstance(moves, str):
            moves_list = [m for m in moves.strip().split() if m]
        else:
            moves_list = list(moves)

        for m in moves_list:
            self.apply_move(m)
        return self

    def get_face(self, face: str) -> List[str]:
        """Return the 9 facelet characters for a specific face (U, R, F, D, L, B)."""
        if face not in FACES:
            raise ValueError(f"Unknown face '{face}'. Must be one of {FACES}")
        idx = FACES.index(face)
        return self._facelets[idx * 9 : (idx + 1) * 9]

    def get_2d_net(self) -> str:
        """Generate a human-readable 2D unfolded cross view of the cube.
        
        Layout:
                 +---+
                 | U |
             +---+---+---+---+
             | L | F | R | B |
             +---+---+---+---+
                 | D |
                 +---+
        """
        u = self.get_face("U")
        l = self.get_face("L")
        f = self.get_face("F")
        r = self.get_face("R")
        b = self.get_face("B")
        d = self.get_face("D")

        lines = []
        lines.append("    +---+")
        for row in range(3):
            lines.append(f"    |{''.join(u[row * 3 : (row + 1) * 3])}|")
        lines.append("+---+---+---+---+")
        for row in range(3):
            l_str = "".join(l[row * 3 : (row + 1) * 3])
            f_str = "".join(f[row * 3 : (row + 1) * 3])
            r_str = "".join(r[row * 3 : (row + 1) * 3])
            b_str = "".join(b[row * 3 : (row + 1) * 3])
            lines.append(f"|{l_str}|{f_str}|{r_str}|{b_str}|")
        lines.append("+---+---+---+---+")
        for row in range(3):
            lines.append(f"    |{''.join(d[row * 3 : (row + 1) * 3])}|")
        lines.append("    +---+")
        return "\n".join(lines)

    def scramble(
        self,
        length: int = 5,
        seed: Optional[int] = None,
        avoid_opposite_redundancy: bool = True,
    ) -> List[str]:
        """Scramble the cube in-place and return the generated move sequence.
        
        Args:
            length: Number of random moves to apply.
            seed: Optional integer seed for reproducibility.
            avoid_opposite_redundancy: If True, eliminates consecutive moves on the same face
                                       and redundant commutations (e.g. U D U).
                                       
        Returns:
            List of move strings applied.
        """
        moves = generate_scramble(
            length=length,
            seed=seed,
            avoid_opposite_redundancy=avoid_opposite_redundancy,
        )
        self.apply_moves(moves)
        return moves

    def solve_shortest_path(
        self, max_depth: int = 7, bidirectional: bool = True
    ) -> BFSSolveResult:
        """Compute the exact shortest solution for the current state using BFS."""
        return solve_bfs(self, max_depth=max_depth, bidirectional=bidirectional)

    def __repr__(self) -> str:
        return f"Cube(state='{self.to_facelet_string()}')"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Cube):
            return self._facelets == other._facelets
        if isinstance(other, str):
            return self.to_facelet_string() == other
        return False

    def __hash__(self) -> int:
        return hash(self.to_facelet_string())


# ---------------------------------------------------------------------------
# Scramble Generator
# ---------------------------------------------------------------------------

def generate_scramble(
    length: int = 5,
    seed: Optional[int] = None,
    avoid_opposite_redundancy: bool = True,
) -> List[str]:
    """Generate a valid, non-redundant random scramble.
    
    Rules for valid non-redundant scrambles:
    1. No consecutive moves on the same face (e.g. no 'U' followed by "U'").
    2. Optional: When two moves are on opposite commuting faces (e.g. U and D),
       a third move cannot immediately repeat either face in the pair (e.g., prevents 'U D U').
       
    Args:
        length: Number of moves in the scramble.
        seed: Optional seed for deterministic generation.
        avoid_opposite_redundancy: Whether to enforce commuting axis rules.
        
    Returns:
        List of Singmaster move strings.
    """
    if length <= 0:
        return []

    rng = random.Random(seed) if seed is not None else random.Random()
    modifiers = ("", "'", "2")
    scramble: List[str] = []

    last_face: Optional[str] = None
    second_last_face: Optional[str] = None

    for _ in range(length):
        valid_faces = [f for f in FACES if f != last_face]
        
        if avoid_opposite_redundancy and last_face is not None and second_last_face is not None:
            # If the last two moves were on opposite commuting faces (e.g., U and D),
            # neither face can be chosen next to avoid sequences like U D U or D U D.
            if OPPOSITE_FACES.get(last_face) == second_last_face:
                valid_faces = [f for f in valid_faces if f != second_last_face]

        chosen_face = rng.choice(valid_faces)
        chosen_mod = rng.choice(modifiers)
        scramble.append(chosen_face + chosen_mod)

        second_last_face = last_face
        last_face = chosen_face

    return scramble


# ---------------------------------------------------------------------------
# Exact Shortest-Path Solver (BFS)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BFSSolveResult:
    """Result of an exact BFS solve."""

    is_solved: bool
    distance: int
    moves: List[str]
    nodes_explored: int
    execution_time_ms: float

    def __str__(self) -> str:
        moves_str = " ".join(self.moves) if self.moves else "(already solved)"
        return (
            f"BFSSolveResult(solved={self.is_solved}, distance={self.distance}, "
            f"moves='{moves_str}', nodes={self.nodes_explored}, time={self.execution_time_ms:.2f}ms)"
        )


def _solve_bfs_unidirectional(
    start_state: str,
    target_state: str,
    max_depth: int = 7,
) -> Tuple[Optional[int], Optional[List[str]], int]:
    """Standard forward Breadth-First Search solver."""
    if start_state == target_state:
        return 0, [], 1

    # queue entries: (state_str, last_face, move_history)
    queue = collections.deque([(start_state, None, [])])
    visited = {start_state: 0}
    nodes = 0

    while queue:
        state, last_face, path = queue.popleft()
        depth = len(path)
        if depth >= max_depth:
            continue

        for move, perm in MOVE_PERMUTATIONS.items():
            face = move[0]
            if face == last_face:
                continue

            next_state = "".join(state[perm[i]] for i in range(54))
            nodes += 1

            if next_state == target_state:
                return depth + 1, path + [move], nodes

            if next_state not in visited:
                visited[next_state] = depth + 1
                queue.append((next_state, face, path + [move]))

    return None, None, nodes


def _solve_bfs_bidirectional(
    start_state: str,
    target_state: str,
    max_depth: int = 7,
) -> Tuple[Optional[int], Optional[List[str]], int]:
    """Meet-in-the-Middle Bidirectional BFS solver.
    
    Explores forward from scrambled state and backward from solved state level-by-level,
    dramatically reducing the search space from O(b^d) to O(b^(d/2)).
    Guaranteed to return the exact shortest path.
    """
    if start_state == target_state:
        return 0, [], 1

    # Visited dictionaries: state -> (depth, last_face, path)
    forward_visited: Dict[str, Tuple[int, Optional[str], List[str]]] = {
        start_state: (0, None, [])
    }
    backward_visited: Dict[str, Tuple[int, Optional[str], List[str]]] = {
        target_state: (0, None, [])
    }

    q_forward = collections.deque([(start_state, None)])
    q_backward = collections.deque([(target_state, None)])

    nodes = 0
    best_dist = float("inf")
    best_path: Optional[List[str]] = None

    depth_f = 0
    depth_b = 0

    while q_forward and q_backward:
        if best_dist <= depth_f + depth_b:
            break

        # Expand whichever queue has fewer frontier nodes
        if len(q_forward) <= len(q_backward) and depth_f < (max_depth + 1) // 2 + 1:
            depth_f += 1
            for _ in range(len(q_forward)):
                state, last_face = q_forward.popleft()
                d, _, path = forward_visited[state]
                if d + 1 + depth_b > best_dist:
                    continue

                for move, perm in MOVE_PERMUTATIONS.items():
                    face = move[0]
                    if face == last_face:
                        continue

                    next_state = "".join(state[perm[i]] for i in range(54))
                    nodes += 1

                    if next_state in backward_visited:
                        b_dist, _, b_path = backward_visited[next_state]
                        total_dist = d + 1 + b_dist
                        full_path = path + [move] + invert_moves(b_path)
                        if total_dist < best_dist:
                            best_dist = total_dist
                            best_path = full_path

                    if next_state not in forward_visited:
                        forward_visited[next_state] = (d + 1, face, path + [move])
                        q_forward.append((next_state, face))
        else:
            depth_b += 1
            for _ in range(len(q_backward)):
                state, last_face = q_backward.popleft()
                d, _, path = backward_visited[state]
                if d + 1 + depth_f > best_dist:
                    continue

                for move, perm in MOVE_PERMUTATIONS.items():
                    face = move[0]
                    if face == last_face:
                        continue

                    next_state = "".join(state[perm[i]] for i in range(54))
                    nodes += 1

                    if next_state in forward_visited:
                        f_dist, _, f_path = forward_visited[next_state]
                        total_dist = f_dist + d + 1
                        full_path = f_path + invert_moves(path + [move])
                        if total_dist < best_dist:
                            best_dist = total_dist
                            best_path = full_path

                    if next_state not in backward_visited:
                        backward_visited[next_state] = (d + 1, face, path + [move])
                        q_backward.append((next_state, face))

        if depth_f + depth_b >= max_depth and best_path is not None:
            break

    if best_path is not None:
        return int(best_dist), best_path, nodes
    return None, None, nodes


def solve_bfs(
    cube_or_state: Union[Cube, str],
    max_depth: int = 7,
    bidirectional: bool = True,
) -> BFSSolveResult:
    """Find the exact shortest path from a given cube state to the solved state using BFS.
    
    Args:
        cube_or_state: A Cube instance or 54-character facelet string.
        max_depth: Maximum search depth (moves). Default is 7.
        bidirectional: If True, uses Meet-in-the-Middle BiBFS (recommended, exponential speedup).
                       If False, uses textbook forward BFS.
                       
    Returns:
        BFSSolveResult containing solved status, shortest integer distance, exact moves,
        nodes explored, and execution time in milliseconds.
    """
    if isinstance(cube_or_state, Cube):
        start_state = cube_or_state.to_facelet_string()
    else:
        Cube._validate_state_string(cube_or_state)
        start_state = cube_or_state

    t0 = time.perf_counter()

    if bidirectional:
        dist, moves, nodes = _solve_bfs_bidirectional(
            start_state, SOLVED_STATE, max_depth=max_depth
        )
    else:
        dist, moves, nodes = _solve_bfs_unidirectional(
            start_state, SOLVED_STATE, max_depth=max_depth
        )

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    if dist is not None and moves is not None:
        return BFSSolveResult(
            is_solved=True,
            distance=dist,
            moves=moves,
            nodes_explored=nodes,
            execution_time_ms=elapsed_ms,
        )
    else:
        return BFSSolveResult(
            is_solved=False,
            distance=-1,
            moves=[],
            nodes_explored=nodes,
            execution_time_ms=elapsed_ms,
        )

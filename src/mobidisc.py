# this module contains several functions
# 1) a function which computes whether a plane curve given by a perm rep is self overlapping
# 2) a function which computes all mobidiscs for a given a multiloop
# 3) a function which computes all unicorn annuli for a given multiloop
import numpy as np

from src.permrep import Multiloop
import src.drawloop as drawloop


# TODO: Some of the functions could be replaced by numpy


# Main functions
def is_self_overlapping(sequence: list[tuple]) -> bool:
    """The function accepts a sequence of points in the form of tuple (x,y) and checks if the multiloop is self-overlapping."""
    sequence = remove_collinear_points(sequence)

    whitney_index = cal_whit(sequence)
    if abs(whitney_index) != 1:
        return False
    # Reverse the sequence if the whitney index is negative
    sequence = sequence[::whitney_index]

    n = len(sequence)
    q = init_table_q(sequence, n)
    fill_table_q(q, sequence, n)
    for i in range(n):
        if q[i][i - 1] == 1:
            # If any q[i][i - 1] is 1, it indicates the loop is self-overlapping
            return True

    return False


def compute_mobidiscs(multiloop: Multiloop) -> list[int]:
    """Compute all mobidiscs for a given multiloop."""
    monogons = find_monogons(multiloop)
    bigons = find_bigons(multiloop)

    return {"monogons": monogons, "bigons": bigons}


def compute_mobidiscs_cnf(
    multiloop: Multiloop,
    loop_to_circles: drawloop.CircleAdjacency,
    monogons_he,
    bigons_he,
) -> list[list[int]]:
    cnf = []

    def add_clause(halfedges):
        clause_set = {loop_to_circles.faces_circles[he] for he in halfedges}
        clauses_seen = set()
        if not clause_set:
            return
        frozen = frozenset(clause_set)
        if frozen in clauses_seen:
            return
        clauses_seen.add(frozen)
        cnf.append(sorted(clause_set))

    for monogon in monogons_he:
        halfedges = [
            face_he
            for face_he in loop_to_circles.faces_circles
            if face_he not in multiloop.inf_face
            and is_in_mobidisc(multiloop, face_he, multiloop.inf_face, 0, monogon)
        ]
        add_clause(halfedges)

    for bigon in bigons_he:
        halfedges = [
            face_he
            for face_he in loop_to_circles.faces_circles
            if face_he not in multiloop.inf_face
            and is_in_mobidisc(multiloop, face_he, multiloop.inf_face, 0, bigon)
        ]
        add_clause(halfedges)
    return cnf


def filter_cnf(mobidiscs_cnf: list[list[int]]) -> list[list[int]]:
    unique_originals = []
    seen = set()
    for clause in mobidiscs_cnf:
        s = frozenset(clause)
        if s not in seen:
            seen.add(s)
            unique_originals.append(clause)

    sets = [set(x) for x in unique_originals]
    filtered = []
    for i, si in enumerate(sets):
        for j, sj in enumerate(sets):
            if i == j:
                continue
            if sj < si:
                break
        else:
            filtered.append(unique_originals[i])
    filtered = sorted(filtered, key=lambda clause: (len(clause), clause))
    return filtered


# Helper functions, intended for internal use only


def is_in_mobidisc(
    multiloop: Multiloop,
    start_halfedge: int,
    targets: list[int],
    special_number: int,
    rotation: tuple[int],
    visited: set[int] = None,
) -> bool:
    """Check if a half-edge is in the mobidisc by running DFS."""

    # TODO: Find correct name for special_number
    def rotation_check(
        start_halfedge: int, end_halfedge: int, rotation: tuple[int]
    ) -> int:
        if start_halfedge in rotation:
            if len(rotation) <= 2:
                return 1 if rotation[0] == start_halfedge else -1
            start_index = rotation.index(start_halfedge)
            if rotation[(start_index + 1) % len(rotation)] == end_halfedge:
                return 1
            if rotation[(start_index - 1) % len(rotation)] == end_halfedge:
                return -1
        return 0

    if visited is None:
        visited = set()

    visited.add(start_halfedge)
    curr = start_halfedge
    neighbors = [curr]
    while multiloop.phi(curr) != start_halfedge:
        curr = multiloop.phi(curr)
        neighbors.append(curr)

    for neighbor in neighbors:
        if neighbor in targets:
            return special_number != 0
        next_halfedge = multiloop.eps(neighbor)
        if next_halfedge not in visited:
            if is_in_mobidisc(
                multiloop,
                next_halfedge,
                targets,
                special_number + rotation_check(neighbor, next_halfedge, rotation),
                rotation,
                visited,
            ):
                return True
    return False


def find_monogons(multiloop: Multiloop) -> list[tuple[int]]:
    monogons = set()
    for cycle in multiloop.tau.cycles:
        for half_edge in cycle:
            this_monogon = multiloop.find_strand_between(half_edge, half_edge)
            if not this_monogon:
                continue
            monogons.add(multiloop.canonicalize_strand(this_monogon))
    return monogons


def find_bigons(multiloop: Multiloop) -> list[tuple[int]]:
    bigons = set()
    for cycle in multiloop.sig.cycles:
        for half_edge in cycle:
            fst_strnd_bigon = []
            sec_strnd_bigon = []
            curr = half_edge
            while True:
                curr = (multiloop.sig * multiloop.sig)(curr)
                fst_strnd_bigon.append(curr)
                curr = multiloop.eps(curr)
                fst_strnd_bigon.append(curr)
                if multiloop.is_samevert(curr, half_edge):
                    break
                for sec_strt in [multiloop.sig(curr), multiloop.sig.inv(curr)]:
                    sec_strnd_bigon = multiloop.find_strand_between(sec_strt, half_edge)
                    if sec_strnd_bigon:
                        bigons.add(
                            multiloop.canonicalize_strand(
                                fst_strnd_bigon + sec_strnd_bigon
                            )
                        )
    return bigons


def remove_collinear_points(
    sequence: list[tuple], tolerance: float = 1e-16
) -> list[tuple]:
    """Remove collinear points from the sequence of points."""
    temp_sequence = []
    for i in range(len(sequence)):
        angle = cal_angle(
            sequence[i - 1],
            sequence[i],
            sequence[(i + 1) % len(sequence)],
        )
        # If the angle is not close to zero, keep the point
        if abs(angle) > tolerance:
            temp_sequence.append(sequence[i])
    return temp_sequence


def cross_product(a: tuple, b: tuple, c: tuple) -> float:
    """Calculate the cross product of vectors ab and ac, where a, b, c are points in 2D space."""
    x1, y1 = a
    x2, y2 = b
    x3, y3 = c
    cross_product_value = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
    return cross_product_value


def is_convex(a: tuple, b: tuple, c: tuple) -> bool:
    """Check if the points a, b, c form a convex angle (if it turns right at b)."""
    is_convex_value = cross_product(a, b, c) < 0
    return is_convex_value


def is_counterclockwise(a: tuple, b: tuple, c: tuple) -> bool:
    """Check if the points a, b, c are in counterclockwise order."""
    # If the angle at b is not convex, then it is counterclockwise
    is_counterclockwise_value = not is_convex(a, b, c)
    return is_counterclockwise_value


def is_appear_counterclockwise(vertices: list[tuple]) -> bool:
    """Check if the indices appear in counterclockwise order around a vertex."""
    i, j, k, l, m = vertices  # l = k+1, m = k-1
    # We allow only one of the checks to fail
    checks = [
        ("is_counterclockwise(k, i, j)", is_counterclockwise(k, i, j)),
        (
            "j == l or is_counterclockwise(k, j, k+1)",
            j == l or is_counterclockwise(k, j, l),
        ),
        ("is_counterclockwise(k, k+1, k-1)", is_counterclockwise(k, l, m)),
        (
            "m == i or is_counterclockwise(k, k-1, i)",
            m == i or is_counterclockwise(k, m, i),
        ),
    ]

    toggle = True
    for description, condition in checks:
        if not condition:
            if not toggle:
                return False
            toggle = False
    return True


def is_intersect_interior(a: tuple, b: tuple, triangle: list[tuple]) -> bool:
    """Check if line segments AB (xb-xa, yb-ya) intersect the interior of the triangle"""

    def is_intersect(a: tuple, b: tuple, c: tuple, d: tuple) -> bool:
        """Check if line segments AB and CD intersect."""
        # Using the cross product to determine if segments intersect
        return (
            cross_product(a, b, c) * cross_product(a, b, d) < 0
            and cross_product(c, d, a) * cross_product(c, d, b) < 0
        )

    for i in range(3):
        s1 = triangle[i]
        s2 = triangle[(i + 1) % 3]
        if is_intersect(a, b, s1, s2):
            return True

    return False


def cal_angle(a: tuple, b: tuple, c: tuple) -> float:
    """Calculate the angle between vector AB and BC (in radians) given three points A, B, and C."""
    tolerance = 1e-8
    vector_ab = np.array(b) - np.array(a)
    vector_bc = np.array(c) - np.array(b)

    # Using the formula: θ = arccos((a ⋅ b) / (|a| |b|)), where θ is the angle, 'a' and 'b' are the vectors, '⋅' represents the dot product, and |a| and |b| are the magnitudes of the vectors.
    dot_product = np.dot(vector_ab, vector_bc)
    mag_ab = np.linalg.norm(vector_ab)
    mag_bc = np.linalg.norm(vector_bc)

    cos_theta = dot_product / (mag_ab * mag_bc)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    theta = np.arccos(cos_theta)

    return theta


def cal_whit(sequence) -> float:
    """Calculate the whitney index of the loop"""
    sign = 1
    angles = []
    n = len(sequence)
    for i in range(n):
        prev = sequence[i - 1]
        targ = sequence[i]
        next = sequence[(i + 1) % n]

        is_conv = is_convex(prev, targ, next)
        sign = 1 if is_conv else -1
        angle = sign * cal_angle(prev, targ, next)
        angles.append(angle)

    whitney_index_raw = sum(angles) / (2 * np.pi)

    whitney_index = round(whitney_index_raw)
    return whitney_index


def segment_intersects_triangle(segments: list[tuple], triangle: list[tuple]) -> bool:
    """Check if any segment intersects the interior of the triangle."""
    for segment in segments:
        a, b = segment
        if is_intersect_interior(a, b, triangle):
            return True
    return False


def init_table_q(sequence: list[tuple], n: int) -> list[list[int]]:
    """Initialize the Q table for the self-overlapping check."""
    if not n:
        n = len(sequence)
    q = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        q[i][(i + 1) % n] = 1
        q[i][(i + 2) % n] = (
            1
            if is_convex(
                sequence[i % n],
                sequence[(i + 1) % n],
                sequence[(i + 2) % n],
            )
            else 0
        )
    return q


def fill_table_q(q: list[list[int]], sequence: list[tuple], n: int) -> None:
    """Fill the Q table based on the conditions for self-overlapping."""
    for length in range(3, n):
        for i in range(n):
            j = (i + length) % n
            for k in range(n):

                if q[i][k % n] != 1 or q[k % n][j % n] != 1:
                    continue

                if not is_counterclockwise(
                    sequence[i],
                    sequence[j % n],
                    sequence[k % n],
                ):
                    continue

                if not is_appear_counterclockwise(
                    [
                        sequence[i],
                        sequence[j % n],
                        sequence[(k) % n],
                        sequence[(k + 1) % n],
                        sequence[(k - 1) % n],
                    ]
                ):
                    continue

                triangle = [
                    sequence[i],
                    sequence[j],
                    sequence[k],
                ]

                segments_to_check = [
                    (sequence[i], sequence[(i + 1) % n]),
                    (sequence[(k - 1) % n], sequence[k]),
                    (sequence[k], sequence[(k + 1) % n]),
                    (sequence[(j - 1) % n], sequence[j]),
                ]

                if segment_intersects_triangle(segments_to_check, triangle):
                    continue

                q[i][j % n] = 1
                break

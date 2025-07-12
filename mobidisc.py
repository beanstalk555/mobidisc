# this module contains several functions
# 1) a function which computes whether a plane curve given by a perm rep is self overlapping
# 2) a function which computes all mobidiscs for a given a multiloop
# 3) a function which computes all unicorn annuli for a given multiloop
from permrep import Multiloop
from circlepack import CirclePack
from drawloop import generate_circles
import logging
import numpy as np

logger = logging.getLogger(__name__)


def cross_product(a: tuple, b: tuple, c: tuple) -> float:
    """Calculate the cross product of vectors AB and AC, where A, B, C are points in 2D space."""
    x1, y1 = a
    x2, y2 = b
    x3, y3 = c
    return (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)


def is_convex(a: tuple, b: tuple, c: tuple) -> bool:
    """Check if the points (x1, y1), (x2, y2), (x3, y3) form a convex angle (if it turns right at (x2, y2))."""
    # Using the cross product to determine the orientation
    return cross_product(a, b, c) < 0


def is_counterclockwise(a: tuple, b: tuple, c: tuple) -> bool:
    """Check if the points (x1, y1), (x2, y2), (x3, y3) are in counterclockwise order."""
    # Using is_convex to determine if the angle at 'c' is convex
    # If the angle is convex, then it is counterclockwise
    return is_convex(a, c, b)


def is_appear_counterclockwise(vertices: list[int]) -> bool:
    """Check if the indices appear in counterclockwise order around a vertex."""
    i, j, k, l, m = vertices  # l = k+1, m = k-1
    toggle = True
    if not is_counterclockwise(k, i, j):
        if not toggle:
            return False
        toggle = False
    if not (j == l) and not is_counterclockwise(k, j, l):
        if not toggle:
            return False
        toggle = False
    if not is_counterclockwise(k, l, m):
        if not toggle:
            return False
        toggle = False
    if not (m == i) and not is_counterclockwise(k, m, i):
        if not toggle:
            return False
        toggle = False
    return True


def is_intersect_interior(a: tuple, b: tuple, triangle: list[tuple]) -> bool:
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
    vector_ab = np.array(b) - np.array(a)
    vector_bc = np.array(c) - np.array(b)
    print(f"Vectors: AB={vector_ab}, BC={vector_bc}")
    dot_product = np.dot(vector_ab, vector_bc)
    mag_ab = np.linalg.norm(vector_ab)
    mag_ac = np.linalg.norm(vector_bc)
    if mag_ab == 0 or mag_ac == 0:
        raise ValueError("One of the vectors has zero length.")
    cos_theta = dot_product / (mag_ab * mag_ac)
    theta = np.arccos(cos_theta)
    return theta


def cal_whit(circle_pack: CirclePack, sequence) -> float:
    """Calculate the whitney index of the loop"""
    sign = 1
    angles = []
    for i in range(len(sequence)):
        if is_convex(
            circle_pack[sequence[i - 1]],
            circle_pack[sequence[i]],
            circle_pack[sequence[(i + 1) % len(sequence)]],
        ):
            sign = 1
        else:
            sign = -1
        print(
            f"Convex check: {sequence[i-1]}, {sequence[i]}, {sequence[(i+1)%len(sequence)]} = {sign}"
        )
        angle = sign * cal_angle(
            circle_pack[sequence[i - 1]],
            circle_pack[sequence[i]],
            circle_pack[sequence[(i + 1) % len(sequence)]],
        )
        print(f"Angle: {angle}")
        angles.append(angle)
    whitney_index = sum(angles) / (2 * np.pi)
    whitney_index = round(whitney_index)
    print(f"Whitney index: {whitney_index}")
    return whitney_index
    # return


def is_self_overlapping(multiloop: "Multiloop") -> bool:

    tolerance = 1e-5  # Tolerance for collinearity checks

    assigned_circles = generate_circles(multiloop)
    packed_circles = CirclePack(
        assigned_circles["internal"], assigned_circles["external"]
    )
    sequences = assigned_circles["sequences"][0]["circle_ids"]
    for circle_id, (center, radius) in packed_circles.items():
        packed_circles[circle_id] = (center.real, center.imag)

    logger.info(f"Checking multiloop with sequences: {sequences}")
    temp_sequences = []
    for i in range(len(sequences)):
        cross_product_result = cross_product(
            packed_circles[sequences[i - 1]],
            packed_circles[sequences[i]],
            packed_circles[sequences[(i + 1) % len(sequences)]],
        )
        if abs(cross_product_result) > tolerance:
            temp_sequences.append(sequences[i])
    sequences = temp_sequences
    logger.info(f"Sequence after removing collinear points: {sequences}")
    whitney_index = cal_whit(packed_circles, sequences)
    if abs(whitney_index) != 1:
        return False
    sequences = sequences[::whitney_index]
    n = len(sequences)
    q = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        q[i][(i + 1) % n] = 1
        q[i][(i + 2) % n] = (
            1
            if is_convex(
                packed_circles[sequences[i % n]],
                packed_circles[sequences[(i + 1) % n]],
                packed_circles[sequences[(i + 2) % n]],
            )
            else 0
        )
    for length in range(3, n):
        for i in range(n):
            j = (i + length) % n
            for k in range(n):
                logger.info(
                    f"Checking i={i}, j={j}, k={k}, circles={sequences[i % n]}, {sequences[j % n]}, {sequences[k % n]}"
                )
                # Check if Q[i][k] = Q[k][j] = 1
                if not q[i][k % n] or not q[k % n][j % n]:
                    logger.info(f"Skipping k={k} due to Q[i][k] or Q[k][j] not being 1")
                    continue
                logger.info(f"Q[i][k] and Q[k][j] are both 1 for k={k}")
                # Check if v[i]v[j]v[k] is oriented counterclockwise
                if not is_counterclockwise(
                    packed_circles[sequences[i % n]],
                    packed_circles[sequences[j % n]],
                    packed_circles[sequences[k % n]],
                ):
                    logger.info(f"Skipping k={k} due to counterclockwise orientation")
                    continue
                logger.info(f"v[i]v[j]v[k] is oriented counterclockwise for k={k}")
                # Check if v[i], v[j], v[k+1], and v[k-1], appear in that order counterclockwise around v[k]
                logger.info(
                    f"Checking appearance order around v[k] for k={k}: {sequences[i % n]}, {sequences[j % n]}, {sequences[k % n]}, {sequences[(k + 1) % n]}, {sequences[(k - 1) % n]}"
                )
                if not is_appear_counterclockwise(
                    [
                        packed_circles[sequences[i % n]],
                        packed_circles[sequences[j % n]],
                        packed_circles[sequences[(k) % n]],
                        packed_circles[sequences[(k + 1) % n]],
                        packed_circles[sequences[(k - 1) % n]],
                    ]
                ):
                    logger.info(f"Skipping k={k} due to appearance order around v[k]")
                    continue
                logger.info(
                    f"v[i], v[j], v[k+1], and v[k-1] appear in counterclockwise order around v[k] for k={k}"
                )
                # Check if the following four segments intersect the interior of v[i]v[j]v[k]: v[i]v[i+1],, v[k-1]v[k], v[k]v[k+1], and v[j-1]v[j]
                if is_intersect_interior(
                    packed_circles[sequences[i % n]],
                    packed_circles[sequences[(i + 1) % n]],
                    [
                        packed_circles[sequences[i % n]],
                        packed_circles[sequences[j % n]],
                        packed_circles[sequences[k % n]],
                    ],
                ):
                    logger.info(f"Skipping k={k} due to intersection with v[i]v[i+1]")
                    continue

                if is_intersect_interior(
                    packed_circles[sequences[(k - 1) % n]],
                    packed_circles[sequences[k % n]],
                    [
                        packed_circles[sequences[i % n]],
                        packed_circles[sequences[j % n]],
                        packed_circles[sequences[k % n]],
                    ],
                ):
                    logger.info(f"Skipping k={k} due to intersection with v[k-1]v[k]")
                    continue

                if is_intersect_interior(
                    packed_circles[sequences[k % n]],
                    packed_circles[sequences[(k + 1) % n]],
                    [
                        packed_circles[sequences[i % n]],
                        packed_circles[sequences[j % n]],
                        packed_circles[sequences[k % n]],
                    ],
                ):
                    logger.info(f"Skipping k={k} due to intersection with v[k]v[k+1]")
                    continue

                if is_intersect_interior(
                    packed_circles[sequences[(j - 1) % n]],
                    packed_circles[sequences[j % n]],
                    [
                        packed_circles[sequences[i % n]],
                        packed_circles[sequences[j % n]],
                        packed_circles[sequences[k % n]],
                    ],
                ):
                    logger.info(f"Skipping k={k} due to intersection with v[j-1]v[j]")
                    continue
                logger.info(f"All condition checks passed for k={k}")
                # If we reach here, it means there exists an index k that satisfies the conditions
                q[i][j % n] = 1
                break

    print("Table Q:")
    for row in q:
        print(row)

    for i in range(n):
        if q[i][i - 1] == 1:
            # If we find a self-overlapping condition, we can return True immediately
            return True

    return False

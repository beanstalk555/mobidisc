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

# TODO: Some of the functions could be replaced by numpy
# TODO: Remove unnecessary logging


def cross_product(a: tuple, b: tuple, c: tuple) -> float:
    """Calculate the cross product of vectors ab and ac, where a, b, c are points in 2D space."""
    x1, y1 = a
    x2, y2 = b
    x3, y3 = c
    logger.debug(
        f"Calculating cross product for vectors: ab=({x2 - x1}, {y2 - y1}), ac=({x3 - x1}, {y3 - y1})"
    )
    cross_product_value = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
    logger.debug(f"Cross product value: {cross_product_value}")
    return cross_product_value


def is_convex(a: tuple, b: tuple, c: tuple) -> bool:
    """Check if the points a, b, c form a convex angle (if it turns right at b)."""
    # Using the cross product to determine the orientation
    is_convex_value = cross_product(a, b, c) < 0
    logger.debug(f"Points a={a}, b={b}, c={c} form a convex angle: {is_convex_value}")
    return is_convex_value


def is_counterclockwise(a: tuple, b: tuple, c: tuple) -> bool:
    """Check if the points a, b, c are in counterclockwise order."""
    # Using is_convex to determine if the angle at 'b' is convex
    # If the angle is not convex, then it is counterclockwise
    is_counterclockwise_value = not is_convex(a, b, c)
    logger.debug(
        f"Points a={a}, b={b}, c={c} are in counterclockwise order: {is_counterclockwise_value}"
    )
    return is_counterclockwise_value


def is_appear_counterclockwise(vertices: list[int]) -> bool:
    """Check if the indices appear in counterclockwise order around a vertex."""
    i, j, k, l, m = vertices  # l = k+1, m = k-1
    logger.debug(
        f"Checking counterclockwise appearance for vertices: i={i}, j={j}, k={k}, k+1={l}, k-1={m}"
    )
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
                logger.debug(f"Second failure at {description}")
                return False
            logger.debug(f"First failure at {description}")
            toggle = False
    logger.debug(
        f"v[i], v[j], v[k+1], and v[k-1] appear in counterclockwise order around v[k] for k={k}"
    )
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

    logger.debug(
        f"Cheking if the line segment AB ({b[0]-a[0]}, {b[1]-a[1]}) intersect the interior of the triangle {triangle}"
    )

    for i in range(3):
        s1 = triangle[i]
        s2 = triangle[(i + 1) % 3]
        if is_intersect(a, b, s1, s2):
            logger.debug(
                f"Segment AB intersects with side ({s2[0]-s1[0]}, {s2[1]-s1[1]})"
            )
            return True

    logger.debug(
        f"The line segment AB ({b[0]-a[0]}, {b[1]-a[1]})does not intersect the interior of the triangle {triangle}"
    )
    return False


def cal_angle(a: tuple, b: tuple, c: tuple) -> float:
    """Calculate the angle between vector AB and BC (in radians) given three points A, B, and C."""
    vector_ab = np.array(b) - np.array(a)
    vector_bc = np.array(c) - np.array(b)
    logger.debug(
        f"Calculating the angle between vectors: AB={vector_ab}, BC={vector_bc}"
    )

    # Using the formula: θ = arccos((a ⋅ b) / (|a| |b|)), where θ is the angle, 'a' and 'b' are the vectors, '⋅' represents the dot product, and |a| and |b| are the magnitudes of the vectors.
    dot_product = np.dot(vector_ab, vector_bc)
    mag_ab = np.linalg.norm(vector_ab)
    mag_ac = np.linalg.norm(vector_bc)
    cos_theta = dot_product / (mag_ab * mag_ac)
    theta = np.arccos(cos_theta)

    logger.debug(f"The angle between vectors AB={vector_ab}, BC={vector_bc} = {theta}")
    return theta


def cal_whit(circle_pack: CirclePack, sequence) -> float:
    """Calculate the whitney index of the loop"""
    sign = 1
    angles = []
    logger.debug(f"Starting Whitney index calculation for sequence: {sequence}")
    n = len(sequence)
    for i in range(n):
        prev = circle_pack[sequence[i - 1]]
        targ = circle_pack[sequence[i]]
        next = circle_pack[sequence[(i + 1) % n]]

        is_conv = is_convex(prev, targ, next)
        sign = 1 if is_conv else -1
        logger.debug(
            f"Convex check at index {i}: ({sequence[i-1]}, {sequence[i]}, {sequence[(i+1)%n]}) -> "
            f"points {prev}, {targ}, {next} => {'convex' if is_conv else 'concave'} (sign={sign})"
        )

        angle = sign * cal_angle(prev, targ, next)
        logger.debug(
            f"Angle at index {i}: signed angle between AB and BC = {angle:.6f} radians"
        )
        angles.append(angle)

    whitney_index_raw = sum(angles) / (2 * np.pi)
    logger.debug(f"Whitney index before rounding: {whitney_index_raw:.6f}")

    whitney_index = round(whitney_index_raw)
    logger.debug(f"Whitney index after rounding: {whitney_index}")
    return whitney_index
    # return


def is_self_overlapping(multiloop: "Multiloop") -> bool:
    tolerance = 1e-5  # Tolerance for collinearity checks

    logger.info("Generating circle assignments for multiloop...")
    assigned_circles = generate_circles(multiloop)
    packed_circles = CirclePack(
        assigned_circles["internal"], assigned_circles["external"]
    )
    sequences = assigned_circles["sequences"][0]["circle_ids"]
    logger.debug(f"Original circle IDs in sequence: {sequences}")

    for circle_id, (center, radius) in packed_circles.items():
        packed_circles[circle_id] = (center.real, center.imag)

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

    logger.debug(f"Sequence after removing collinear points: {sequences}")
    logger.info(
        f"Checking if multiloop with sequences: {sequences} is self-overlapping"
    )
    whitney_index = cal_whit(packed_circles, sequences)
    logger.info(f"Computed Whitney index: {whitney_index}")

    if abs(whitney_index) != 1:
        logger.info("Whitney index is not +1 or -1, loop is not self-overlapping.")
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
                logger.debug(
                    f"Checking i={i}, j={j}, k={k} (circles: {sequences[i]}, {sequences[j]}, {sequences[k]})"
                )

                # Check if Q[i][k] = Q[k][j] = 1
                if not q[i][k % n] or not q[k % n][j % n]:
                    logger.debug(f"Skipping k={k}: q[i][k] or q[k][j] is 0")
                    continue

                # Check if v[i]v[j]v[k] is oriented counterclockwise
                if not is_counterclockwise(
                    packed_circles[sequences[i % n]],
                    packed_circles[sequences[j % n]],
                    packed_circles[sequences[k % n]],
                ):
                    logger.debug(f"Skipping k={k}: not counterclockwise")
                    continue

                # Check if v[i], v[j], v[k+1], and v[k-1], appear in that order counterclockwise around v[k]
                if not is_appear_counterclockwise(
                    [
                        packed_circles[sequences[i % n]],
                        packed_circles[sequences[j % n]],
                        packed_circles[sequences[(k) % n]],
                        packed_circles[sequences[(k + 1) % n]],
                        packed_circles[sequences[(k - 1) % n]],
                    ]
                ):
                    logger.debug(f"Skipping k={k}: appearance order check failed")
                    continue

                # Check if the following four segments intersect the interior of v[i]v[j]v[k]: v[i]v[i+1],, v[k-1]v[k], v[k]v[k+1], and v[j-1]v[j]
                triangle = [
                    packed_circles[sequences[i]],
                    packed_circles[sequences[j]],
                    packed_circles[sequences[k]],
                ]

                segments_to_check = [
                    (sequences[i], sequences[(i + 1) % n]),
                    (sequences[(k - 1) % n], sequences[k]),
                    (sequences[k], sequences[(k + 1) % n]),
                    (sequences[(j - 1) % n], sequences[j]),
                ]

                intersect_flag = False
                for a_idx, b_idx in segments_to_check:
                    if is_intersect_interior(
                        packed_circles[a_idx], packed_circles[b_idx], triangle
                    ):
                        logger.debug(
                            f"Skipping k={k}: segment ({a_idx}, {b_idx}) intersects interior"
                        )
                        intersect_flag = True
                        break

                if intersect_flag:
                    continue
                logger.debug(
                    f"All conditions passed for i={i}, j={j}, k={k}, marking q[i][j] = 1"
                )
                # If we reach here, it means there exists an index k that satisfies the conditions
                q[i][j % n] = 1
                break

    logger.debug("Final Q table:")
    for row in q:
        logger.debug(row)

    for i in range(n):
        if q[i][i - 1] == 1:
            # If any q[i][i - 1] is 1, it indicates the loop is self-overlapping
            logger.info(f"q[{i}][{(i - 1) % n}] = 1 => The loop is self-overlapping.")
            return True

    logger.info(
        "There is no index i such that Q[i][i-1] = 1, hence the loop is not self-overlapping"
    )
    return False

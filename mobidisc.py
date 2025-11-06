# this module contains several functions
# 1) a function which computes whether a plane curve given by a perm rep is self overlapping
# 2) a function which computes all mobidiscs for a given a multiloop
# 3) a function which computes all unicorn annuli for a given multiloop
import logging
import numpy as np
from logging_utils.logger import setup_logger
from permrep import Multiloop

logger = setup_logger(
    "mobidisc_logger",
    "logs/mobidisc.log",
    level=logging.DEBUG,
)

# TODO: Some of the functions could be replaced by numpy
# TODO: Remove unnecessary logging


# Main functions
def is_self_overlapping(sequence: list[tuple]) -> bool:
    """The function accepts a sequence of points in the form of tuple (x,y) and checks if the multiloop is self-overlapping."""
    logger.info(f"Checking if multiloop with sequence: {sequence} is self-overlapping")
    sequence = remove_collinear_points(sequence)
    logger.info(f"Sequence after removing collinear points: {sequence}")

    whitney_index = cal_whit(sequence)
    if abs(whitney_index) != 1:
        logger.info("Whitney index is not +1 or -1, loop is not self-overlapping.")
        return False
    # Reverse the sequence if the whitney index is negative
    sequence = sequence[::whitney_index]

    n = len(sequence)
    q = init_table_q(sequence, n)
    fill_table_q(q, sequence, n)
    for i in range(n):
        if q[i][i - 1] == 1:
            # If any q[i][i - 1] is 1, it indicates the loop is self-overlapping
            logger.info(f"q[{i}][{(i - 1) % n}] = 1 => The loop is self-overlapping.")
            return True

    logger.info(
        "There is no index i such that Q[i][i-1] = 1, hence the loop is not self-overlapping"
    )
    return False


def compute_mobidiscs(multiloop: Multiloop) -> list[int]:
    """Compute all mobidiscs for a given multiloop."""
    logger.info(f"Computing mobidiscs for multiloop: {multiloop}")
    monogons = find_monogons(multiloop)
    bigons = find_bigons(multiloop)

    return {"monogons": monogons, "bigons": bigons}


def compute_mobidiscs_cnf(multiloop: Multiloop) -> list[tuple[int]]:
    mobidiscs = compute_mobidiscs(multiloop)
    print("Monogons:", mobidiscs["monogons"])
    print("Bigons:", mobidiscs["bigons"])


# Helper functions, intended for internal use only


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
                curr = multiloop.tau(curr)
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
    logger.debug(f"Sequence after removing collinear points: {temp_sequence}")
    return temp_sequence


def cross_product(a: tuple, b: tuple, c: tuple) -> float:
    """Calculate the cross product of vectors ab and ac, where a, b, c are points in 2D space."""
    x1, y1 = a
    x2, y2 = b
    x3, y3 = c
    logger.debug(
        f"Calculating cross product for vectors: ({x2 - x1}, {y2 - y1}), ({x3 - x1}, {y3 - y1})"
    )
    cross_product_value = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
    logger.debug(f"Cross product value: {cross_product_value}")
    return cross_product_value


def is_convex(a: tuple, b: tuple, c: tuple) -> bool:
    """Check if the points a, b, c form a convex angle (if it turns right at b)."""
    is_convex_value = cross_product(a, b, c) < 0
    logger.debug(f"Points a={a}, b={b}, c={c} form a convex angle: {is_convex_value}")
    return is_convex_value


def is_counterclockwise(a: tuple, b: tuple, c: tuple) -> bool:
    """Check if the points a, b, c are in counterclockwise order."""
    # If the angle at b is not convex, then it is counterclockwise
    is_counterclockwise_value = not is_convex(a, b, c)
    logger.debug(
        f"Points a={a}, b={b}, c={c} are in counterclockwise order: {is_counterclockwise_value}"
    )
    return is_counterclockwise_value


def is_appear_counterclockwise(vertices: list[tuple]) -> bool:
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
    tolerance = 1e-8
    vector_ab = np.array(b) - np.array(a)
    vector_bc = np.array(c) - np.array(b)
    logger.debug(
        f"Calculating the angle between vectors: AB={vector_ab}, BC={vector_bc}"
    )

    # Using the formula: θ = arccos((a ⋅ b) / (|a| |b|)), where θ is the angle, 'a' and 'b' are the vectors, '⋅' represents the dot product, and |a| and |b| are the magnitudes of the vectors.
    dot_product = np.dot(vector_ab, vector_bc)
    mag_ab = np.linalg.norm(vector_ab)
    mag_bc = np.linalg.norm(vector_bc)

    cos_theta = dot_product / (mag_ab * mag_bc)
    theta = np.arccos(cos_theta)

    logger.debug(f"The angle between vectors AB={vector_ab}, BC={vector_bc} = {theta}")
    return theta


def cal_whit(sequence) -> float:
    """Calculate the whitney index of the loop"""
    sign = 1
    angles = []
    logger.debug(f"Starting Whitney index calculation for sequence: {sequence}")
    n = len(sequence)
    for i in range(n):
        prev = sequence[i - 1]
        targ = sequence[i]
        next = sequence[(i + 1) % n]

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
    logger.info(f"Whitney index before rounding: {whitney_index_raw:.6f}")

    whitney_index = round(whitney_index_raw)
    logger.debug(f"Whitney index after rounding: {whitney_index}")
    logger.info(f"Computed Whitney index: {whitney_index}")
    return whitney_index


def segment_intersects_triangle(segments: list[tuple], triangle: list[tuple]) -> bool:
    """Check if any segment intersects the interior of the triangle."""
    for segment in segments:
        a, b = segment
        if is_intersect_interior(a, b, triangle):
            logger.debug(f"Segment {segment} intersects the interior of the triangle.")
            return True
    logger.debug("No segments intersect the interior of the triangle.")
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
                logger.debug(
                    f"Checking i={i}, j={j}, k={k} (circles: {sequence[i]}, {sequence[j]}, {sequence[k]})"
                )

                if q[i][k % n] != 1 or q[k % n][j % n] != 1:
                    logger.debug(f"Skipping k={k}: q[i][k] or q[k][j] is 0")
                    continue

                if not is_counterclockwise(
                    sequence[i],
                    sequence[j % n],
                    sequence[k % n],
                ):
                    logger.debug(f"Skipping k={k}: not counterclockwise")
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
                    logger.debug(f"Skipping k={k}: appearance order check failed")
                    continue

                # Check if the following four segments intersect the interior of v[i]v[j]v[k]: v[i]v[i+1],, v[k-1]v[k], v[k]v[k+1], and v[j-1]v[j]
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

                logger.debug(
                    f"All conditions passed for i={i}, j={j}, k={k}, marking q[i][j] = 1"
                )
                # If we reach here, it means there exists an index k that satisfies the conditions
                q[i][j % n] = 1
                break

    logger.debug("Final Q table:")
    for row in q:
        logger.debug(row)

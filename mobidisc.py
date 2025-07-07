# this module contains several functions
# 1) a function which computes whether a plane curve given by a perm rep is self overlapping
# 2) a function which computes all mobidiscs for a given a multiloop
# 3) a function which computes all unicorn annuli for a given multiloop
from permrep import Multiloop
from circlepack import CirclePack
from drawloop import generate_circles


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


def is_appear_counterclockwise(indices: list[int]) -> bool:
    """Check if the indices appear in counterclockwise order around a vertex."""
    i, j, k, l = indices
    print("Function is_appear_counterclockwise is not implemented yet.")
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


def is_self_overlapping(multiloop: "Multiloop") -> bool:
    assigned_circles = generate_circles(multiloop)
    packed_circles = CirclePack(
        assigned_circles["internal"], assigned_circles["external"]
    )
    sequences = assigned_circles["sequences"][0]["circle_ids"]
    for circle_id, (center, radius) in packed_circles.items():
        packed_circles[circle_id] = (center.real, center.imag)

    print(sequences)

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
                # Check if Q[i][k] = Q[k][j] = 1
                if not q[i][k % n] or not q[k % n][j % n]:
                    continue
                # Check if v[i]v[j]v[k] is oriented counterclockwise
                if not is_counterclockwise(
                    packed_circles[sequences[i % n]],
                    packed_circles[sequences[j % n]],
                    packed_circles[sequences[k % n]],
                ):
                    continue

                # Check if v[i], v[j], v[k+1], and v[k-1], appear in that order counterclockwise around v[k]
                if not is_appear_counterclockwise([i, j, k - 1, k + 1]):
                    continue

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
                    continue

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

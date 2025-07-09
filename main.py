import permrep as perm
import ranloop
import drawloop
from circlepack import CirclePack
from mobidisc import is_self_overlapping

if __name__ == "__main__":
    # example_loop = ranloop.generate_planar(10)
    example_loop = perm.Multiloop([[-4, -3, 1, -1], [3, 4, -2, 2]])
    example_loop.inf_face = [-2]
    # example_loop = perm.Multiloop(
    #     [
    #         (9, 14, -10, -1),
    #         (7, 2, -8, -3),
    #         (3, 6, -4, -7),
    #         (11, 4, -12, -5),
    #         (1, 8, -2, -9),
    #         (13, 10, -14, -11),
    #         (5, 12, -6, -13),
    #     ]
    # )
    # example_loop.inf_face = (-2, 7, -4, 11, -14, 9)

    loop_to_circles = drawloop.generate_circles(example_loop)
    packed_circles = CirclePack(
        loop_to_circles["internal"], loop_to_circles["external"]
    )
    print(example_loop)
    drawloop.drawloop(packed_circles, sequences=loop_to_circles["sequences"], scale=500)

    print(is_self_overlapping(example_loop))

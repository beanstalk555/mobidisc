import permrep as perm
import ranloop
import drawloop
from circlepack import CirclePack

if __name__ == "__main__":
    # Vertex permutation
    # multiloop = [[-1, -4, -3, -2], [1, 2, 6, 5], [3, 8, 7, -6], [-8, 4, -5, -7]]
    # example_loop = perm.Multiloop(multiloop)
    # example_loop.inf_face = [-1, 5, 4]
    multiloop = [
        (6, 1, -7, -2),
        (13, 2, -14, -3),
        (11, 4, -12, -5),
        (5, 10, -6, -11),
        (16, 7, -1, -8),
        (14, 9, -15, -10),
        (3, 12, -4, -13),
        (8, 15, -9, -16),
    ]
    example_loop = perm.Multiloop(multiloop)
    example_loop.inf_face =  (-1,6,10,-15,8)

    # TODO: Fix the problem where some randomly generated loop doesn't work with the circle packing algorithm
    print(example_loop)
    loop_to_circles = drawloop.generate_circles(example_loop, example_loop.inf_face)
    print(
        f"{loop_to_circles["internal"]}\n{loop_to_circles["external"]}\n{loop_to_circles["sequences"]}"
    )
    # drawloop.drawloop(
    #     CirclePack(loop_to_circles[0], loop_to_circles[1]),
    #     loop_to_circles[2][0],
    # )
    # print(CirclePack(loop_to_circles[0], loop_to_circles[1]), loop_to_circles[2][0])
    drawloop.drawloop(
        CirclePack(loop_to_circles["internal"], loop_to_circles["external"]),
        filename="circle_pack.svg",
        sequence=loop_to_circles["sequences"][0],
    )

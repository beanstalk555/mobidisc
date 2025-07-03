import permrep as perm
import ranloop
import drawloop
from circlepack import CirclePack, InvertAround

if __name__ == "__main__":
    # Vertex permutation
    # multiloop = [
    #     (13, 4, -14, -5),
    #     (1, 6, -2, -7),
    #     (17, 12, -18, -13),
    #     (3, 14, -4, -15),
    #     (15, 2, -16, -3),
    #     (5, 16, -6, -17),
    #     (11, 18, -12, -9),
    #     (8, 9, -1, -10),
    #     (10, 7, -11, -8),
    # ]
    # example_loop = perm.Multiloop(multiloop)
    # example_loop.inf_face = (-2,15,-4,13,-18,11,7)
    # multiloop = [[-1, -4, -3, -2], [1, 2, 6, 5], [3, 8, 7, -6], [-8, 4, -5, -7]]
    # example_loop = perm.Multiloop(multiloop)
    # example_loop.inf_face = [-1, 5, 4]

    # multiloop = [[-4, 4, 1, 3], [-2, 2, -1, -3]]
    # example_loop = perm.Multiloop(multiloop)
    # example_loop.inf_face = [4, 3, -1]
    # multiloop = [(-2, -3, 4, -4), (1, -1, 3, 2)]
    # example_loop = perm.Multiloop(multiloop)
    # example_loop.inf_face = [4]
    example_loop = ranloop.generate_planar(4)

    print(example_loop)
    loop_to_circles = drawloop.generate_circles(example_loop, example_loop.inf_face)
    print(
        f"{loop_to_circles["internal"]}\n{loop_to_circles["external"]}\n{loop_to_circles["sequences"]}"
    )
    drawloop.drawloop(
        CirclePack(loop_to_circles["internal"], loop_to_circles["external"]),
        filename="circle_pack.svg",
        sequences=loop_to_circles["sequences"],
    )
    # internal = {
    #     1: [4, 10, 3, 5, 11, 6],
    #     7: [11, 5, 2, 12, 8],
    #     8: [12, 2, 6, 11, 7],
    #     10: [1, 4, 3],
    #     11: [1, 5, 7, 8, 6],
    #     12: [2, 8, 7],
    # }
    # external = {3: 1, 4: 1, 6: 1, 5: 1, 2: 1}
    # sequences = [
    #     [4, -4, 3, 4, 1, -4, 5, -1, 2, -1, 7, 2, 8, -2, 2, 2, 6, 3, 1, 3],
    #     [5, 1, 1, 1, 3, 4, 4, -4, 1, 4, 6, -3, 2, -3, 8, -2, 7, 2, 2, -2],
    # ]
    # drawloop.drawloop(
    #     CirclePack(internal, external),
    #     filename="circle_pack.svg",
    #     sequences=sequences,
    # )

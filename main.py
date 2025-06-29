import permrep as perm
import ranloop
import drawloop
from circlepack import CirclePack

if __name__ == "__main__":
    # Vertex permutation
    multiloop = [[-1, -4, -3, -2], [1, 2, 6, 5], [3, 8, 7, -6], [-8, 4, -5, -7]]
    example_loop = perm.Multiloop(multiloop)
    example_loop.inf_face = [-1, 5, 4]
    # multiloop = [(-2, -3, 4, -4), (1, -1, 3, 2)]
    # example_loop = perm.Multiloop(multiloop)
    # example_loop.inf_face = [3,-2]
    # example_loop = ranloop.generate_planar(8)

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

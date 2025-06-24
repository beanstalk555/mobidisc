import permrep as perm
import ranloop
import drawloop
from circlepack import CirclePack

if __name__ == "__main__":
    # Vertex permutation
    multiloop = [[-1, -4, -3, -2], [1, 2, 6, 5], [3, 8, 7, -6], [-8, 4, -5, -7]]
    # multiloop = [
    #     [1,-1,2,-2]
    # ]
    example_loop = perm.Multiloop(multiloop)
    # example_loop.inf_face = [-1,-2]
    example_loop.inf_face = [-1, 5, 4]

    # TODO: Fix the problem where some randomly generated loop doesn't work with the circle packing algorithm
    print(example_loop)
    loop_to_circles = drawloop.generate_circles(example_loop, example_loop.inf_face)
    print(f"{loop_to_circles[0]}\n{loop_to_circles[1]}\n{loop_to_circles[2]}")
    drawloop.drawloop(
        CirclePack(loop_to_circles[0], loop_to_circles[1]),
        loop_to_circles[2][0],
    )

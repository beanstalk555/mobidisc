import permrep as perm
import ranloop
import drawloop
from circlepack import CirclePack

if __name__ == "__main__":
    # Vertex permutation
    # multiloop = [
    #     [-1,-2,-3,-4],
    #     [2, 1, 5, 6],
    #     [3, -6, 7, 8],
    #     [-5, 4, -8, -7]
    # ]
    multiloop = [
        [10, 1, -11, -2],
        [3, 8, -4, -9],
        [4, 11, -5, -12],
        [12, 5, -1, -6],
        [9, 6, -10, -7],
        [7, 2, -8, -3],
    ]
    example_loop = perm.Multiloop(multiloop)
    example_loop.inf_face = [-4,-12,-6,9]
    # example_loop.inf_face = [1, -4, -5]

    # TODO: Fix the problem where some randomly generated loop doesn't work with the circle packing algorithm
    a = True
    while a:
        a = False
        try:
            # random_multiloop = ranloop.generate_planar(3)
            # example_loop = random_multiloop
            # print(example_loop)

            loop_to_circles = drawloop.generate_circles(
                example_loop, example_loop.inf_face
            )
            print(f"{loop_to_circles[0]}\n{loop_to_circles[1]}\n{loop_to_circles[2]}")
            drawloop.drawloop(
                CirclePack(loop_to_circles[0], loop_to_circles[1]),
                loop_to_circles[2][0],
            )
        except:
            a = True

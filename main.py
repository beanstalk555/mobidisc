import permrep as perm
import ranloop
import drawloop
from circlepack import CirclePack
from mobidisc import is_self_overlapping

if __name__ == "__main__":
    # example_loop = ranloop.generate_planar(2)
    example_loop = perm.Multiloop([[-4, -3, 1, -1], [3, 4, -2, 2]])
    example_loop.inf_face = [-2]

    loop_to_circles = drawloop.generate_circles(example_loop)
    packed_circles = CirclePack(
        loop_to_circles["internal"], loop_to_circles["external"]
    )
    print(example_loop)
    drawloop.drawloop(packed_circles, sequences=loop_to_circles["sequences"], scale=500)

    print(is_self_overlapping(example_loop))

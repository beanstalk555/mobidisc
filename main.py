import permrep as perm
import ranloop
import drawloop
from circlepack import CirclePack, InvertAround

if __name__ == "__main__":
    example_loop = ranloop.generate_planar(8)

    print(example_loop)
    loop_to_circles = drawloop.generate_circles(example_loop, example_loop.inf_face)
    print(
        f"{loop_to_circles["internal"]}\n{loop_to_circles["external"]}\n{loop_to_circles["sequences"]}"
    )
    drawloop.drawloop(
        CirclePack(loop_to_circles["internal"], loop_to_circles["external"]),
        filename="circle_pack.svg",
        sequences=loop_to_circles["sequences"],
        scale=500
    )

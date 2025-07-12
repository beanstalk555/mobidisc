import permrep as perm
import ranloop
import drawloop
from circlepack import CirclePack
from mobidisc import is_self_overlapping
import logging

logger = logging.getLogger(__name__)


def init_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        filename="mobidisc.log",
        filemode="w",
    )


if __name__ == "__main__":
    init_logging()
    # [[-3, -4, 2, -2], [3, 1, -1, 4]]
    # [2]
    example_loop = ranloop.generate_planar(2)
    print(example_loop)
    loop_to_circles = drawloop.generate_circles(example_loop)
    packed_circles = CirclePack(
        loop_to_circles["internal"], loop_to_circles["external"]
    )
    print(f"Sequences: {loop_to_circles['sequences']}")
    drawloop.drawloop(
        packed_circles,
        sequences=loop_to_circles["sequences"],
        scale=500,
        withLabels=True,
        filename=f"test_loops/loop_test.svg",
    )
    print(is_self_overlapping(example_loop))
    # example_loop = perm.Multiloop(
    #     [
    #         [1, 8, -2, -9],
    #         [4, -10, -5, 9],
    #         [5, 12, -6, -13],
    #         [36, -14, -1, 13],
    #         [6, -16, -7, 15],
    #         [3, 20, -4, -21],
    #         [2, -22, -3, 21],
    #         [11, 26, -12, -27],
    #         [18, -26, -19, 25],
    #         [10, -28, -11, 27],
    #         [17, -31, -18, 30],
    #         [19, 28, -20, -29],
    #         [24, -30, -25, 29],
    #         [16, 31, -17, -32],
    #         [23, 32, -24, -33],
    #         [22, -34, -23, 33],
    #         [7, 34, -8, -35],
    #         [14, -36, -15, 35],
    #     ]
    # )
    # for i in range(len(example_loop.phi)):
    #     example_loop.inf_face = example_loop.phi[
    #         i
    #     ]
    #     loop_to_circles = drawloop.generate_circles(example_loop)
    #     packed_circles = CirclePack(
    #         loop_to_circles["internal"], loop_to_circles["external"]
    #     )
    #     print(example_loop)
    #     drawloop.drawloop(
    #         packed_circles,
    #         sequences=loop_to_circles["sequences"],
    #         scale=500,
    #         filename=f"test_loops/reverse_loop_{i}_{is_self_overlapping(example_loop)}.svg",
    #     )

        # print(is_self_overlapping(example_loop))

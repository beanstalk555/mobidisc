import permrep as perm
import ranloop
import drawloop
from circlepack import CirclePack
from mobidisc import build_matrix
import logging
import numpy as np

logger = logging.getLogger(__name__)

def init_logging():
    logging.basicConfig(
        level=logging.INFO,
        filename="mobidisc.log",
        filemode="w",
        encoding='utf-8'
    )

if __name__ == "__main__":
    init_logging()

    #example_loop = ranloop.generate_planar(6)
    example_loop = ranloop.Multiloop([[-12, 11, -8, 10], [-10, -4, 4, 12], [6, -6, -2, -7], [-3, 3, -5, 8], [1, 5, -9, -1], [7, 2, 9, -11]])
    example_loop.inf_face = [11, 9, 5, 3, 8]
    print(example_loop)
    loop_to_circles = drawloop.generate_circles(example_loop)
    packed_circles = CirclePack(
        loop_to_circles["internal"], loop_to_circles["external"]
    )
    print(f"Sequences: {loop_to_circles['sequences']}")
    drawloop.drawloop(
        packed_circles,
        sequences=loop_to_circles["sequences"],
        withLabels=True,
        filename=f"test_loops/loop_test.svg",
     )
    
    matrix = build_matrix(example_loop, loop_to_circles["sequences"], loop_to_circles["assignments"], packed_circles)
    print(matrix)

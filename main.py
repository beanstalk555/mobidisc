import permrep as perm
import ranloop
import drawloop
from circlepack import CirclePack
from mobidisc import is_self_overlapping
import logging

logger = logging.getLogger(__name__)


def init_logging():
    logging.basicConfig(
        level=logging.INFO, filename="mobidisc.log", filemode="w", encoding="utf-8"
    )


def preprocess_multiloop(multiloop: perm.Multiloop):
    """Preprocess the multiloop to draw and check for self-overlapping."""
    loop_to_circles = drawloop.generate_circles(multiloop)
    packed_circles = CirclePack(
        loop_to_circles["internal"], loop_to_circles["external"]
    )
    pcir_to_coordinates = {}
    for circle_id, (center, radius) in packed_circles.items():
        pcir_to_coordinates[circle_id] = (center.real, center.imag)
    sequences_of_coor = []
    for i in range(len(loop_to_circles["sequences"])):
        sequences_of_coor.append(
            [
                pcir_to_coordinates[j]
                for j in loop_to_circles["sequences"][i]["circle_ids"]
            ]
        )
    return {
        "PackedCircles": packed_circles,
        "SequencesOfCircles": loop_to_circles["sequences"],
        "SequencesOfCoordinations": sequences_of_coor,
    }


def main():
    init_logging()
    example_loop = ranloop.generate_planar(2)
    print(example_loop)

    proccessed_loop = preprocess_multiloop(example_loop)
    print(proccessed_loop)
    strand_no = 0
    sequence_of_coordinations = proccessed_loop["SequencesOfCoordinations"][strand_no]
    print(is_self_overlapping(sequence_of_coordinations))


if __name__ == "__main__":
    main()

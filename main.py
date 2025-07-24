import permrep as perm
import ranloop
import drawloop
from circlepack import CirclePack
from mobidisc import is_self_overlapping, compute_mobidiscs
import logging
from logging_utils.logger import setup_logger


def preprocess_multiloop(multiloop: perm.Multiloop):
    """Preprocess the multiloop to draw and check for self-overlapping."""
    loop_to_circles = drawloop.CircleAdjacency(multiloop)
    packed_circles = CirclePack(loop_to_circles.internal, loop_to_circles.external)
    pcir_to_coordinates = {}
    for circle_id, (center, radius) in packed_circles.items():
        pcir_to_coordinates[circle_id] = (center.real, center.imag)
    sequences_of_coor = []
    
    for i in range(len(loop_to_circles.sequences)):
        sequences_of_coor.append(
            [pcir_to_coordinates[j] for j in loop_to_circles.sequences[i]["circle_ids"]]
        )
    return {
        "PackedCircles": packed_circles,
        "SequencesOfCircles": loop_to_circles.sequences,
        "SequencesOfCoordinations": sequences_of_coor,
    }


def main():
    logger = setup_logger(
        "main_logger",
        "logs/main.log",
        level=logging.INFO,
    )

    example_loop = ranloop.generate_planar(4)
    logger.info(f"Generated multiloop: {example_loop}")

    proccessed_loop = preprocess_multiloop(example_loop)
    logger.info(f"Processed info from the loop: {proccessed_loop}")

    strand_no = 0
    sequence_of_coordinations = proccessed_loop["SequencesOfCoordinations"][strand_no]
    logger.info(
        f"Is the sequence of coordinations self-overlapping?: {is_self_overlapping(sequence_of_coordinations)}"
    )

    logger.info(
        f"Drawing the loop with sequence {proccessed_loop['SequencesOfCircles']}"
    )
    drawloop.drawloop(
        sequences=proccessed_loop["SequencesOfCircles"],
        circle_dict=proccessed_loop["PackedCircles"],
        showCircLabels=True,
        filename="loops/loop.svg",
    )

    logger.info(f"Mobidiscs: {compute_mobidiscs(example_loop)}")


if __name__ == "__main__":
    main()

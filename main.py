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

    monogons = multiloop.find_monogons()
    monogons_to_circles = loop_to_circles.build_sequences(monogons)
    sequences_monogons_coor = []
    for i in range(len(monogons_to_circles)):
        sequences_monogons_coor.append(
            [pcir_to_coordinates[j] for j in monogons_to_circles[i]["circle_ids"]]
        )
    filtered_monogon_coors = []
    filtered_monogon_circles = []
    for i in range(len(sequences_monogons_coor)):
        if is_self_overlapping(sequences_monogons_coor[i]):
            filtered_monogon_coors.append(sequences_monogons_coor[i])
            filtered_monogon_circles.append(monogons_to_circles[i])

    bigons = multiloop.find_bigons()
    bigons_to_circles = loop_to_circles.build_sequences(bigons)
    sequences_bigons_coor = []
    for i in range(len(bigons_to_circles)):
        sequences_bigons_coor.append(
            [pcir_to_coordinates[j] for j in bigons_to_circles[i]["circle_ids"]]
        )
    filtered_bigon_coors = []
    filtered_bigon_circles = []
    for i in range(len(sequences_bigons_coor)):
        if is_self_overlapping(sequences_bigons_coor[i]):
            filtered_bigon_coors.append(sequences_bigons_coor[i])
            filtered_bigon_circles.append(bigons_to_circles[i])
    
    
    return {
        "PackedCircles": packed_circles,
        "SequencesOfCircles": loop_to_circles.sequences,
        "SequencesOfCoordinations": sequences_of_coor,
        "SequencesOfMonogons": filtered_monogon_circles,
        "SequencesOfMonogonsCoordinations": filtered_monogon_coors,
        "SequencesOfBigons": filtered_bigon_circles,
        "SequencesOfBigonsCoordinations": filtered_bigon_coors,
    }


def main():
    logger = setup_logger(
        "main_logger",
        "logs/main.log",
        level=logging.INFO,
    )

    example_loop = ranloop.generate_planar(3)
    # example_loop = perm.Multiloop(
    #     [[6, -5, -2, 2], [1, -6, 3, -1], [-3, -4, 4, 5]], [-2]
    # )
    logger.info(f"Generated multiloop: {example_loop}")

    proccessed_loop = preprocess_multiloop(example_loop)
    logger.debug(f"Processed info from the loop: {proccessed_loop}")

    strand_no = 0
    sequence_of_coordinations = proccessed_loop["SequencesOfCoordinations"][strand_no]
    logger.info(
        f"Is the sequence of coordinations self-overlapping?: {is_self_overlapping(sequence_of_coordinations)}"
    )

    logger.info(
        f"Drawing the loop with sequence {proccessed_loop['SequencesOfCircles']}"
    )
    drawloop.DrawLoop(
        sequences=proccessed_loop["SequencesOfCircles"],
        circle_dict=proccessed_loop["PackedCircles"],
        showCircLabels=True,
        filename="loops/loop.svg",
    )
    drawloop.DrawLoop(
        sequences=proccessed_loop["SequencesOfMonogons"],
        circle_dict=proccessed_loop["PackedCircles"],
        showCircLabels=True,
        showEdgeLabels=False,
        filename="loops/loop_monogons.svg",
    )
    drawloop.DrawLoop(
        sequences=proccessed_loop["SequencesOfBigons"],
        circle_dict=proccessed_loop["PackedCircles"],
        showCircLabels=True,
        showEdgeLabels=False,
        filename="loops/loop_bigons.svg",
    )
    mobidiscs = compute_mobidiscs(example_loop)
    logger.info(f"Mobidiscs: {mobidiscs}")


if __name__ == "__main__":
    main()

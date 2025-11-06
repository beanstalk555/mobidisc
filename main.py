import permrep as perm
import ranloop
import drawloop
from circlepack import CirclePack
from mobidisc import is_self_overlapping, compute_mobidiscs, compute_mobidiscs_cnf
import logging
from logging_utils.logger import setup_logger


class MobidiscProcessor:
    def __init__(self, multiloop: perm.Multiloop):
        self.multiloop = multiloop
        self.loop_to_circles = drawloop.CircleAdjacency(self.multiloop)
        self.packed_circles = CirclePack(
            self.loop_to_circles.internal, self.loop_to_circles.external
        )
        self.circle_coord = self.build_circle_coord()

        self.main_sequence = self.loop_to_circles.sequences
        self.main_sequence_coord = self.sequences_to_coord(self.main_sequence)

        self.mobidiscs = compute_mobidiscs(self.multiloop)
        self.monogons = self.mobidiscs["monogons"]
        self.monogons_circles, self.monogons_coord = self.filter_mobidiscs(
            self.monogons
        )
        self.bigons = self.mobidiscs["bigons"]
        self.bigons_circles, self.bigons_coord = self.filter_mobidiscs(self.bigons)

    def build_circle_coord(self):
        pcir_to_coordinates = {}
        for circle_id, (center, radius) in self.packed_circles.items():
            pcir_to_coordinates[circle_id] = (center.real, center.imag)
        return pcir_to_coordinates

    def sequences_to_coord(self, sequences):
        res = []
        for i in range(len(sequences)):
            res.append([self.circle_coord[j] for j in sequences[i]["circle_ids"]])
        return res

    def filter_mobidiscs(self, mobidiscs: list[tuple[int]]) -> list[tuple[int]]:
        mobidiscs_circles = self.loop_to_circles.build_sequences(mobidiscs)
        mobidiscs_coord = []
        for i in range(len(mobidiscs_circles)):
            mobidiscs_coord.append(
                [self.circle_coord[j] for j in mobidiscs_circles[i]["circle_ids"]]
            )
        filtered_mobidiscs_coords = []
        filtered_mobidiscs_circles = []
        for i in range(len(mobidiscs_coord)):
            if is_self_overlapping(mobidiscs_coord[i]):
                filtered_mobidiscs_coords.append(mobidiscs_coord[i])
                filtered_mobidiscs_circles.append(mobidiscs_circles[i])
        return filtered_mobidiscs_circles, filtered_mobidiscs_coords


def main():
    logger = setup_logger(
        "main_logger",
        "logs/main.log",
        level=logging.INFO,
    )

    # example_loop = ranloop.generate_planar(3)
    # example_loop = perm.Multiloop(
    #     [[6, -5, -2, 2], [1, -6, 3, -1], [-3, -4, 4, 5]], [-2]
    # )
    example_loop = perm.Multiloop(
        [
            (6, 1, -7, -2),
            (4, 9, -5, -10),
            (10, 5, -1, -6),
            (2, 7, -3, -8),
            (8, 3, -9, -4),
        ],
        (-2, -8, -4, -10, -6),
    )
    logger.info(f"Generated multiloop: {example_loop}")
    compute_mobidiscs_cnf(example_loop)

    proccessed_loop = MobidiscProcessor(example_loop)
    logger.debug(f"Processed info from the loop: {proccessed_loop}")
    logger.info(f"Drawing the loop with sequence {proccessed_loop.main_sequence}")
    drawloop.DrawLoop(
        sequences=proccessed_loop.main_sequence,
        circle_dict=proccessed_loop.packed_circles,
        showCircLabels=True,
        filename="loops/loop.svg",
    )
    drawloop.DrawLoop(
        sequences=proccessed_loop.monogons_circles,
        circle_dict=proccessed_loop.packed_circles,
        showCircLabels=True,
        showEdgeLabels=False,
        filename="loops/loop_monogons.svg",
    )
    drawloop.DrawLoop(
        sequences=proccessed_loop.bigons_circles,
        circle_dict=proccessed_loop.packed_circles,
        showCircLabels=True,
        showEdgeLabels=False,
        filename="loops/loop_bigons.svg",
    )


if __name__ == "__main__":
    main()

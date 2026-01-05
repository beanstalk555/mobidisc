import datetime
import permrep as perm
import drawloop as drawloop
from circlepack import CirclePack
from mobidisc import (
    filter_cnf,
    is_self_overlapping,
    compute_mobidiscs,
    compute_mobidiscs_cnf,
)


class MobidiscProcessor:
    def __init__(self, multiloop: perm.Multiloop):
        self.multiloop = multiloop
        self.mobidiscs_cnf = []
        for face in sorted(multiloop.phi.cycles, key=lambda x: len(x)):
            self.multiloop.inf_face = face
            self.loop_to_circles = drawloop.CircleAdjacency(self.multiloop)
            self.face_circles = set(self.loop_to_circles.faces_circles.values())
            self.packed_circles = CirclePack(
                self.loop_to_circles.internal, self.loop_to_circles.external
            )
            self.circle_coord = self.build_circle_coord()

            self.main_sequence = self.loop_to_circles.sequences
            self.main_sequence_coord = self.sequences_to_coord(self.main_sequence)

            self.mobidiscs = compute_mobidiscs(self.multiloop)

            self.monogons = self.mobidiscs["monogons"]
            self.monogons_circles, self.monogons_coord, self.monogons_he = (
                self.filter_mobidiscs(self.monogons)
            )

            self.bigons = self.mobidiscs["bigons"]
            self.bigons_circles, self.bigons_coord, self.bigons_he = (
                self.filter_mobidiscs(self.bigons)
            )

            self.mobidiscs_cnf.extend(
                compute_mobidiscs_cnf(
                    self.multiloop,
                    self.loop_to_circles,
                    self.monogons_he,
                    self.bigons_he,
                )
            )
        self.mobidiscs_cnf = filter_cnf(self.mobidiscs_cnf)

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

    def mobidiscs_to_tau_cycles(self, mobidiscs: list[tuple[int]]):
        tau_cycles = []
        for mobidisc in mobidiscs:
            tau_cycle = []
            if len(mobidisc) <= 2:
                tau_cycles.append([mobidisc[1]])
                continue
            for i in range(len(mobidisc)):
                half_edge = mobidisc[i]
                if half_edge == -mobidisc[(i + 1) % len(mobidisc)]:
                    continue
                tau_cycle.append(half_edge)
            tau_cycles.append(tau_cycle)
        return tau_cycles

    def filter_mobidiscs(self, mobidiscs: list[tuple[int]]):
        mobidiscs = list(mobidiscs)
        mobidiscs_circles = self.loop_to_circles.build_sequences(
            self.mobidiscs_to_tau_cycles(mobidiscs)
        )
        mobidiscs_coord = self.sequences_to_coord(mobidiscs_circles)
        filtered_mobidiscs_coords = []
        filtered_mobidiscs_circles = []
        filtered_mobidiscs_he = []
        for i in range(len(mobidiscs_coord)):
            if is_self_overlapping(mobidiscs_coord[i]):
                filtered_mobidiscs_coords.append(mobidiscs_coord[i])
                filtered_mobidiscs_circles.append(mobidiscs_circles[i])
                filtered_mobidiscs_he.append(mobidiscs[i])
        return (
            filtered_mobidiscs_circles,
            filtered_mobidiscs_coords,
            filtered_mobidiscs_he,
        )

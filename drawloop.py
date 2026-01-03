"""draws a planar multiloop from a permutation representation with given unbounded region using the circle packing algorithm"""

import svgwrite.path
from permrep import Multiloop
import matplotlib.pyplot as plt
import numpy as np
import svgwrite
from math import sqrt
import os
from circlepack import CirclePack


class CircleAdjacency:
    """A class to hold the circle assignments for vertices, edges, and faces."""

    def __init__(self, multiloop: Multiloop):
        self.multiloop = multiloop

        self.vertices_circles: dict[int, int] = {}
        self.edges_circles: dict[int, int] = {}
        self.faces_circles: dict[int, int] = {}
        self.circle_id = 1
        self.assign_face_circles(multiloop.phi.cycles)
        self.assign_vertex_circles(multiloop.sig.cycles)
        self.assign_edge_circles(multiloop.eps.cycles, multiloop)

        self.internal: dict[int, list[int]] = {}
        self._initialize_internal_dict()
        self.external: dict[int, int] = {}

        self.sequences = []
        self.circle_adjacency = self.build_circle_adjacency()

    def _initialize_internal_dict(self) -> None:
        """Initializes the internal dictionary with empty lists for each circle."""
        max_circle = self.circle_id
        for i in range(1, max_circle):
            self.internal[i] = []

    def assign_vertex_circles(self, cycles: list[list[int]]) -> None:
        """Assigns circle IDs incrementally to vertices."""
        for cycle in cycles:
            for he in cycle:
                self.vertices_circles[he] = self.circle_id
            self.circle_id += 1

    def assign_edge_circles(self, cycles: list[list[int]], multiloop) -> None:
        """Assigns circle IDs incrementally to edges."""
        for cycle in cycles:
            self.edges_circles[cycle[0]] = self.circle_id

            # If both half-edges of the cycle are in the same vertex, it's a monogon.
            # In this case, we need to assign two circle IDs to the edge.
            if multiloop.is_samevert(cycle[0], cycle[1]):
                self.circle_id += 1
            self.edges_circles[cycle[1]] = self.circle_id
            self.circle_id += 1

    def assign_face_circles(self, cycles: list[list[int]]) -> None:
        """Assigns circle IDs incrementally to faces."""
        # Sort faces by sum(2**he) for deterministic ordering, but place the infinite face last if present.
        sort_key = lambda cycle: sum(2 ** abs(he) for he in cycle)
        sorted_cycles = sorted(cycles, key=sort_key)

        for cycle in sorted_cycles:
            for he in cycle:
                self.faces_circles[he] = self.circle_id
            self.circle_id += 1

    def _add_neighbor(self, neighbors: list[int], neighbor: int) -> list[int]:
        """Adds a neighbor to the list if it is not already present."""
        if neighbor not in neighbors:
            neighbors.append(neighbor)
        return neighbors

    def _toggle_neighbor(self, neighbors: list[int], neighbor: int) -> list[int]:
        """If a neighbor is already in the list, delete it."""
        if neighbor not in neighbors:
            neighbors.append(neighbor)
        else:
            neighbors.remove(neighbor)
        return neighbors

    def build_vertex_adjacencies(self) -> None:
        """Builds the adjacency relationships for vertices."""
        for vert_he, vertex_circle in self.vertices_circles.items():
            edge_circle = self.edges_circles[vert_he]
            face_circle = self.faces_circles[vert_he]

            self.internal[vertex_circle].append(edge_circle)

            self.internal[vertex_circle] = self._toggle_neighbor(
                self.internal[vertex_circle], face_circle
            )

    def build_face_adjacencies(self) -> None:
        """Builds the adjacency relationships for faces."""
        for face_he, face_circle in self.faces_circles.items():
            vertex_circle = self.vertices_circles[face_he]
            edge_circle = self.edges_circles[face_he]
            opposite_edge_circle = self.edges_circles[-face_he]
            self.internal[face_circle] = self._toggle_neighbor(
                self.internal[face_circle], vertex_circle
            )

            self.internal[face_circle] = self._add_neighbor(
                self.internal[face_circle], edge_circle
            )
            self.internal[face_circle] = self._add_neighbor(
                self.internal[face_circle], opposite_edge_circle
            )

    def build_edge_adjacencies(self) -> None:
        """Builds the adjacency relationships for edges."""
        for edge_he, edge_circle in self.edges_circles.items():
            face_circle = self.faces_circles[edge_he]
            opposite_face_circle = self.faces_circles[-edge_he]
            vertex_circle = self.vertices_circles[edge_he]

            self.internal[edge_circle] = self._add_neighbor(
                self.internal[edge_circle], face_circle
            )

            if face_circle not in self.internal[vertex_circle]:
                sig_edge_circle = self.edges_circles[self.multiloop.sig(edge_he)]
                self.internal[edge_circle] = self._add_neighbor(
                    self.internal[edge_circle], sig_edge_circle
                )

            self.internal[edge_circle] = self._add_neighbor(
                self.internal[edge_circle], vertex_circle
            )

            if opposite_face_circle not in self.internal[vertex_circle]:
                sig_inv_edge_circle = self.edges_circles[
                    self.multiloop.sig.inv(edge_he)
                ]
                self.internal[edge_circle] = self._add_neighbor(
                    self.internal[edge_circle], sig_inv_edge_circle
                )

            opposite_edge_circle = self.edges_circles[-edge_he]
            if opposite_edge_circle != edge_circle:
                self.internal[edge_circle] = self._add_neighbor(
                    self.internal[edge_circle], opposite_face_circle
                )
                self.internal[edge_circle] = self._add_neighbor(
                    self.internal[edge_circle], opposite_edge_circle
                )

    def process_infinite_face(self, inf_face: list[int]) -> None:
        """Processes the infinite face to update internal and external circle relationships."""
        for inf_he in inf_face:
            face_circle = self.faces_circles[inf_he]
            edge_circle = self.edges_circles[inf_he]
            opposite_edge_circle = self.edges_circles[-inf_he]
            vertex_circle = self.vertices_circles[inf_he]

            self.internal.pop(face_circle, None)
            self.internal.pop(edge_circle, None)
            self.internal.pop(opposite_edge_circle, None)

            self.external[edge_circle] = 1
            self.external[opposite_edge_circle] = 1

            try:
                if face_circle in self.internal[vertex_circle]:
                    self.internal.pop(vertex_circle, None)
                    self.external[vertex_circle] = 1
            except KeyError:
                pass

    def build_sequences(self, tau_cycles: list[list[int]]) -> list[list[int]]:
        """Builds sequences of circle assignments for the given strands (tau) cycles."""
        sequences = []
        visited = set()
        for strand in tau_cycles:
            sequence = {"circle_ids": [], "half_edges": []}
            for i in range(len(strand)):
                he = strand[i]
                edge_circle = self.edges_circles[he]
                opposite_edge_circle = self.edges_circles[-he]
                vertex_circle = self.vertices_circles[he]

                if opposite_edge_circle != edge_circle:
                    sequence["circle_ids"].append(opposite_edge_circle)

                sequence["circle_ids"].append(edge_circle)

                sequence["circle_ids"].append(vertex_circle)
                sequence["half_edges"].append((vertex_circle, he))
            sequence["circle_ids"] = self.multiloop.canonicalize_strand(sequence["circle_ids"])
            if sequence["circle_ids"] not in visited:
                sequences.append(sequence)
                visited.add(sequence["circle_ids"])
        return sequences

    def build_circle_adjacency(self):
        multiloop = self.multiloop
        inf_face = multiloop.inf_face

        self.build_face_adjacencies()
        self.build_vertex_adjacencies()
        self.build_edge_adjacencies()
        self.process_infinite_face(inf_face)

        self.sequences = self.build_sequences(multiloop.tau.cycles)

        # return {
        #     "internal": self.internal,
        #     "external": self.external,
        #     "sequences": self.sequences,
        # }


class DrawLoop:
    def __init__(
        self,
        sequences,
        circle_dict,
        filename,
        showCircLabels=False,
        showEdgeLabels=False,
        showCirc=False,
        scale=200,
        padding=50,
        circle_color="blue",
        circle_stroke_weight=1.0,
        sequence_colors=[
            "red",
            "blue",
            "green",
            "cyan",
            "magenta",
            "orange",
            "darkgreen",
            "brown",
            "pink",
            "navy",
            "purple",
        ],
        sequence_stroke_width=10,
        circle_label_font_scale=300,
        edge_label_font_size=10,
        cnf_data=None,
        cnf_font_size=50,
        infCircLabel=None,
    ):
        self.sequences = sequences
        self.circle_dict = circle_dict
        self.filename = filename
        self.showCircLabels = showCircLabels
        self.showEdgeLabels = showEdgeLabels
        self.infCircLabel = infCircLabel
        self.showCirc = showCirc
        self.scale = scale
        self.padding = padding
        
        # CNF display
        self.cnf_data = cnf_data
        self.cnf_font_size = cnf_font_size

        # Color and style configuration
        self.circle_color = circle_color
        self.circle_stroke_weight = circle_stroke_weight
        self.sequence_colors = sequence_colors
        self.sequence_stroke_width = sequence_stroke_width

        # Font size configuration
        self.circle_label_font_scale = circle_label_font_scale
        self.edge_label_font_size = edge_label_font_size

        self.dwg = None
        self.to_svg_coords = None

        if not isinstance(filename, (str, bytes, os.PathLike)):
            raise TypeError(f"Filename must be a string or path, got {type(filename)}")

        # Determine bounds
        self.dwg = self.setup_canvas()

        # Draw circles (and labels if requested)
        if self.showCirc:
            self.drawcircles()
        if self.showCircLabels:
            # Draw labels even when circles are hidden
            self.drawlabels()

        # Draw sequences
        self.drawsequences()

        # Draw CNF if provided
        if self.cnf_data:
            self.draw_cnf()

        self.dwg.save()

    def setup_canvas(self):
        """Sets up the canvas for drawing."""
        min_x = min(z.real - r for z, r in self.circle_dict.values())
        max_x = max(z.real + r for z, r in self.circle_dict.values())
        min_y = min(z.imag - r for z, r in self.circle_dict.values())
        max_y = max(z.imag + r for z, r in self.circle_dict.values())

        span_x = (max_x - min_x) * self.scale
        span_y = (max_y - min_y) * self.scale

        # Reserve space for CNF at bottom-right
        self.cnf_lines = []
        if self.cnf_data:
            self.cnf_lines = [
                "(" + " ∨ ".join(str(lit) for lit in clause) + ")"
                for clause in self.cnf_data
            ]
        cnf_margin = 8
        cnf_block_height = (
            (len(self.cnf_lines)) * self.cnf_font_size * 1.1 + cnf_margin
            if self.cnf_lines
            else 0
        )
        cnf_block_width = cnf_margin  # small horizontal margin; text is right-aligned

        width = span_x + cnf_block_width
        height = span_y + cnf_block_height

        dwg = svgwrite.Drawing(self.filename, size=(width, height))
        dwg.viewbox(0, 0, width, height)

        # Add white background
        dwg.add(dwg.rect(insert=(0, 0), size=(width, height), fill="white"))

        def to_svg_coords(center):
            cx = (center.real - min_x) * self.scale
            cy = (max_y - center.imag) * self.scale
            return (cx, cy)

        # stash for CNF placement
        self._canvas_width = width
        self._canvas_height = height
        self._cnf_margin = cnf_margin
        self.to_svg_coords = to_svg_coords
        self._min_x = min_x
        self._max_y = max_y
        return dwg

    def drawcircles(self):
        for name, (center, radius) in self.circle_dict.items():
            cx, cy = self.to_svg_coords(center)
            r = radius * self.scale
            self.dwg.add(
                self.dwg.circle(
                    center=(cx, cy),
                    r=r,
                    fill="none",
                    stroke=self.circle_color,
                    stroke_width=radius * self.circle_stroke_weight,
                )
            )
            self.dwg.add(
                self.dwg.circle(
                    center=(cx, cy),
                    r=(radius),
                    fill=self.circle_color,
                    stroke=self.circle_color,
                    stroke_width=30 * radius * self.circle_stroke_weight,
                )
            )

    def drawlabels(self):
        if self.infCircLabel is not None:
            self.circle_dict[self.infCircLabel] = (
                complex(
                    (self._min_x + 1),
                    (self._max_y - 1),
                ),
                0.5,
            )
        for name, (center, radius) in self.circle_dict.items():
            if (
                isinstance(self.showCircLabels, (list, set, tuple))
                and name not in self.showCircLabels
            ):
                continue
            cx, cy = self.to_svg_coords(center)
            self.dwg.add(
                self.dwg.text(
                    str(name),
                    insert=(cx, cy),
                    fill="black",
                    font_size=f"{self.circle_label_font_scale*radius}px",
                    text_anchor="middle",
                    alignment_baseline="central",
                )
            )
        

    def draw_cnf(self):
        """Renders CNF formula at bottom-right corner of the SVG."""
        if not self.cnf_data:
            return

        width = self._canvas_width
        height = self._canvas_height

        margin = self._cnf_margin
        x_pos = width - margin
        # stack from bottom up with spacing
        total_height = (len(self.cnf_lines)) * self.cnf_font_size * 1.1
        y_start = height - margin - total_height

        # Clauses with ∧ between
        for i, line in enumerate(self.cnf_lines):
            y_pos = y_start + (i + 1) * self.cnf_font_size * 1.1
            self.dwg.add(
                self.dwg.text(
                    line + (" ∧" if i < len(self.cnf_lines) - 1 else ""),
                    insert=(x_pos, y_pos),
                    fill="black",
                    font_size=f"{self.cnf_font_size}px",
                    text_anchor="end",
                    font_family="monospace",
                )
            )

    def drawsequences(self):
        tolerance = 1e-2  # how accurately to approximate things
        for seq_index, sequence in enumerate(self.sequences):
            # Get color for this sequence (repeats if more sequences than colors)
            sequence_color = self.sequence_colors[seq_index % len(self.sequence_colors)]

            circle_ids = sequence["circle_ids"]
            for i in range(0, len(circle_ids)):
                startcirc_center = self.circle_dict[circle_ids[i]][0]
                startcirc_r = self.circle_dict[circle_ids[i]][1] * self.scale
                targetcirc_center = self.circle_dict[
                    circle_ids[(i + 1) % len(circle_ids)]
                ][0]
                targetcirc_r = (
                    self.circle_dict[circle_ids[(i + 1) % len(circle_ids)]][1]
                    * self.scale
                )

                endcirc_center = self.circle_dict[
                    circle_ids[(i + 2) % len(circle_ids)]
                ][0]
                endcirc_r = (
                    self.circle_dict[circle_ids[(i + 2) % len(circle_ids)]][1]
                    * self.scale
                )

                def find_intersection(c1, r1, c2, r2):
                    x1 = c1[0]
                    x2 = c2[0]
                    y1 = c1[1]
                    y2 = c2[1]
                    d = r1 + r2
                    return ((x1 + (r1 * (x2 - x1)) / d), (y1 + (r1 * (y2 - y1)) / d))

                start = self.to_svg_coords(startcirc_center)
                target = self.to_svg_coords(targetcirc_center)
                end = self.to_svg_coords(endcirc_center)
                s_curve = find_intersection(start, startcirc_r, target, targetcirc_r)
                e_curve = find_intersection(target, targetcirc_r, end, endcirc_r)
                if self.showEdgeLabels:
                    curr_circ = circle_ids[(i + 1) % len(circle_ids)]
                    if (
                        sequence["half_edges"]
                        and curr_circ == sequence["half_edges"][0][0]
                    ):
                        self.dwg.add(
                            self.dwg.text(
                                str(sequence["half_edges"][0][1]),
                                insert=s_curve,
                                fill="red",
                                font_size=f"{self.edge_label_font_size}px",
                                text_anchor="middle",
                                alignment_baseline="middle",
                            )
                        )
                        sequence["half_edges"].pop(0)

                try:
                    d_bet_se = sqrt(
                        (s_curve[0] - e_curve[0]) ** 2 + (s_curve[1] - e_curve[1]) ** 2
                    )

                    if (
                        abs((d_bet_se / 2) - targetcirc_r) < tolerance
                    ):  # In case 3 circles are on the same line.
                        self.dwg.add(
                            self.dwg.line(
                                start=s_curve,
                                end=e_curve,
                                stroke=sequence_color,
                                stroke_width=self.sequence_stroke_width,
                            )
                        )
                        continue
                    r_curve = sqrt(
                        ((d_bet_se / 2) ** 2 * targetcirc_r**2)
                        / (targetcirc_r**2 - (d_bet_se / 2) ** 2)
                    )

                except ValueError:
                    raise ValueError(
                        f"Error finding the curve between start {s_curve} and end {e_curve} points. Half of the distance between them is {d_bet_se/2}, target circle radius is {targetcirc_r}."
                    )
                arc_path = svgwrite.path.Path(
                    d=f"M {s_curve[0]},{s_curve[1]}",
                    fill="none",
                    stroke=sequence_color,
                    stroke_width=self.sequence_stroke_width,
                )

                # Use the 'A' command for a circular arc
                arc_path.push_arc(
                    target=e_curve,  # End point of the arc
                    rotation=0,  # No rotation
                    r=r_curve,  # Radius of the arc
                    large_arc=False,  # Large arc flag (False because we want a minor arc)
                    angle_dir=(
                        "+"
                        if (s_curve[0] - e_curve[0]) * (target[1] - e_curve[1])
                        - (s_curve[1] - e_curve[1]) * (target[0] - e_curve[0])
                        > 0
                        else "-"
                    ),  # Sweep direction, use cross product to determine
                    absolute=True,  # Absolute coordinates
                )

                self.dwg.add(arc_path)

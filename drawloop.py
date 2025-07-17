# draws a planar multiloop from a permutation representation with given unbounded region using the circle packing algorithm
import svgwrite.path
from permrep import Multiloop
import matplotlib.pyplot as plt
import numpy as np
import svgwrite
from cmath import phase
from math import cos, sin, sqrt
import os

from typing import TypedDict
from dataclasses import dataclass


class CircleAssignments:
    """A class to hold the circle assignments for vertices, edges, and faces."""

    def __init__(self):
        self.vertices: dict[int, int] = {}
        self.edges: dict[int, int] = {}
        self.faces: dict[int, int] = {}
        self.next_circle_id = 1

    def assign_vertex_circles(self, cycles: list[list[int]]) -> None:
        """Assigns circle IDs incrementally to vertices."""
        for cycle in cycles:
            for he in cycle:
                self.vertices[he] = self.next_circle_id
            self.next_circle_id += 1

    def assign_edge_circles(self, cycles: list[list[int]], multiloop) -> None:
        """Assigns circle IDs incrementally to edges."""
        for cycle in cycles:
            self.edges[cycle[0]] = self.next_circle_id

            # If both half-edges of the cycle are in the same vertex, it's a monogon.
            # In this case, we need to assign two circle IDs to the edge.
            if multiloop.is_samevert(cycle[0], cycle[1]):
                self.next_circle_id += 1
            self.edges[cycle[1]] = self.next_circle_id
            self.next_circle_id += 1

    def assign_face_circles(self, cycles: list[list[int]]) -> None:
        """Assigns circle IDs incrementally to faces."""
        for cycle in cycles:
            for he in cycle:
                self.faces[he] = self.next_circle_id
            self.next_circle_id += 1


class CircleResult(TypedDict):
    """A typed dictionary to hold the results of circle adjacencies."""

    internal: dict[int, list[int]]
    external: dict[int, int]
    sequences: list[list[int]]
    assignments: CircleAssignments


class AdjacencyBuilder:
    """A class to build the adjacency relationships between circles."""

    def __init__(self, multiloop: "Multiloop", circles: CircleAssignments):
        self.multiloop = multiloop
        self.circles = circles
        self.internal: dict[int, list[int]] = {}
        self._initialize_internal_dict()
        self.external: dict[int, int] = {}

    def _initialize_internal_dict(self) -> None:
        """Initializes the internal dictionary with empty lists for each circle."""
        max_circle = self.circles.next_circle_id
        for i in range(1, max_circle):
            self.internal[i] = []

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

    def build_face_adjacencies(self) -> None:
        """Builds the adjacency relationships for faces."""
        for face_he, face_circle in self.circles.faces.items():
            vertex_circle = self.circles.vertices[face_he]
            edge_circle = self.circles.edges[face_he]
            opposite_edge_circle = self.circles.edges[-face_he]

            self.internal[face_circle] = self._toggle_neighbor(
                self.internal[face_circle], vertex_circle
            )

            self.internal[face_circle] = self._add_neighbor(
                self.internal[face_circle], edge_circle
            )
            self.internal[face_circle] = self._add_neighbor(
                self.internal[face_circle], opposite_edge_circle
            )

    def build_vertex_adjacencies(self) -> None:
        """Builds the adjacency relationships for vertices."""
        for vert_he, vertex_circle in self.circles.vertices.items():
            edge_circle = self.circles.edges[vert_he]
            face_circle = self.circles.faces[vert_he]

            self.internal[vertex_circle].append(edge_circle)

            self.internal[vertex_circle] = self._toggle_neighbor(
                self.internal[vertex_circle], face_circle
            )

    def build_edge_adjacencies(self) -> None:
        """Builds the adjacency relationships for edges."""
        for edge_he, edge_circle in self.circles.edges.items():
            face_circle = self.circles.faces[edge_he]
            opposite_face_circle = self.circles.faces[-edge_he]
            vertex_circle = self.circles.vertices[edge_he]

            self.internal[edge_circle] = self._add_neighbor(
                self.internal[edge_circle], face_circle
            )

            if face_circle not in self.internal[vertex_circle]:
                sig_edge_circle = self.circles.edges[self.multiloop.sig(edge_he)]
                self.internal[edge_circle] = self._add_neighbor(
                    self.internal[edge_circle], sig_edge_circle
                )

            self.internal[edge_circle] = self._add_neighbor(
                self.internal[edge_circle], vertex_circle
            )

            if opposite_face_circle not in self.internal[vertex_circle]:
                sig_inv_edge_circle = self.circles.edges[
                    self.multiloop.sig.inv(edge_he)
                ]
                self.internal[edge_circle] = self._add_neighbor(
                    self.internal[edge_circle], sig_inv_edge_circle
                )

            opposite_edge_circle = self.circles.edges[-edge_he]
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
            face_circle = self.circles.faces[inf_he]
            edge_circle = self.circles.edges[inf_he]
            opposite_edge_circle = self.circles.edges[-inf_he]
            vertex_circle = self.circles.vertices[inf_he]

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


def build_sequences(
    circles: "CircleAssignments", tau_cycles: list[list[int]]
) -> list[list[int]]:
    """Builds sequences of circle assignments for the given strands (tau) cycles."""
    sequences = []

    for strand in tau_cycles:
        sequence = {"circle_ids": [], "half_edges": []}
        for he in strand:
            edge_circle = circles.edges[he]
            opposite_edge_circle = circles.edges[-he]
            vertex_circle = circles.vertices[he]

            if opposite_edge_circle != edge_circle:
                sequence["circle_ids"].append(opposite_edge_circle)
                sequence["half_edges"].append(-he)

            sequence["circle_ids"].append(edge_circle)
            sequence["half_edges"].append(he)

            sequence["circle_ids"].append(vertex_circle)
            sequence["half_edges"].append(he)

        sequences.append(sequence)
    return sequences


def generate_circles(multiloop: "Multiloop") -> CircleResult:
    inf_face = multiloop.inf_face
    circles = CircleAssignments()
    circles.assign_vertex_circles(multiloop.sig.cycles)
    circles.assign_edge_circles(multiloop.eps.cycles, multiloop)
    circles.assign_face_circles(multiloop.phi.cycles)

    adj_builder = AdjacencyBuilder(multiloop, circles)
    adj_builder.build_face_adjacencies()
    adj_builder.build_vertex_adjacencies()
    adj_builder.build_edge_adjacencies()
    adj_builder.process_infinite_face(inf_face)

    sequences = build_sequences(circles, multiloop.tau.cycles)

    return {
        "internal": adj_builder.internal,
        "external": adj_builder.external,
        "sequences": sequences,
        "assignments": circles
    }
    

# act as struct

@dataclass
class svg_globals:
    min_x: float
    max_y: float
    scale: int = 200
    padding: int = 50
    
def to_svg_coords(center):
        cx = (center.real - svg_globals.min_x) * svg_globals.scale + svg_globals.padding
        cy = (svg_globals.max_y - center.imag) * svg_globals.scale + svg_globals.padding
        return (cx, cy)

def drawloop(
    circle_dict,
    filename="circle_pack.svg",
    sequences=None,
    withLabels=False
):
    tolerance = 1e-2  # how accurately to approximate things
    if not isinstance(filename, (str, bytes, os.PathLike)):
        raise TypeError(f"Filename must be a string or path, got {type(filename)}")

    # Determine bounds
    svg_globals.min_x = min(z.real - r for z, r in circle_dict.values())
    max_x = max(z.real + r for z, r in circle_dict.values())
    min_y = min(z.imag - r for z, r in circle_dict.values())
    svg_globals.max_y = max(z.imag + r for z, r in circle_dict.values())

    width = (max_x - svg_globals.min_x) * svg_globals.scale + 2 * svg_globals.padding
    height = (svg_globals.max_y- min_y) * svg_globals.scale + 2 * svg_globals.padding

    dwg = svgwrite.Drawing(filename, size=(width, height))
    dwg.viewbox(0, 0, width, height)

    # Add white background
    dwg.add(dwg.rect(insert=(0, 0), size=(width, height), fill="white"))

    # Draw circles
    for name, (center, radius) in circle_dict.items():
        cx, cy = to_svg_coords(center)
        r = radius * svg_globals.scale
        dwg.add(
            dwg.circle(
                center=(cx, cy), r=r, fill="none", stroke="black", stroke_width=radius
            )
        )
        dwg.add(
            dwg.circle(
                center=(cx, cy),
                r=(radius),
                fill="black",
                stroke="black",
                stroke_width=30 * radius,
            )
        )
        if withLabels:
            dwg.add(
                dwg.text(
                    str(name),
                    insert=(cx + r * 0.1, cy),
                    fill="black",
                    font_size=f"{45*radius+10}px",
                    text_anchor="middle",
                )
            )

    # Draw connection lines (black, thick)
    if sequences:
        for sequence in sequences:
            circle_ids = sequence["circle_ids"]
            for i in range(0, len(circle_ids)):
                startcirc_center = circle_dict[circle_ids[i]][0]
                startcirc_r = circle_dict[circle_ids[i]][1] * svg_globals.scale
                targetcirc_center = circle_dict[circle_ids[(i + 1) % len(circle_ids)]][
                    0
                ]
                targetcirc_r = (
                    circle_dict[circle_ids[(i + 1) % len(circle_ids)]][1] * svg_globals.scale
                )

                endcirc_center = circle_dict[circle_ids[(i + 2) % len(circle_ids)]][0]
                endcirc_r = (
                    circle_dict[circle_ids[(i + 2) % len(circle_ids)]][1] * svg_globals.scale
                )

                def find_intersection(c1, r1, c2, r2):
                    x1 = c1[0]
                    x2 = c2[0]
                    y1 = c1[1]
                    y2 = c2[1]
                    d = r1 + r2
                    return ((x1 + (r1 * (x2 - x1)) / d), (y1 + (r1 * (y2 - y1)) / d))

                start = to_svg_coords(startcirc_center)
                target = to_svg_coords(targetcirc_center)
                end = to_svg_coords(endcirc_center)
                s_curve = find_intersection(start, startcirc_r, target, targetcirc_r)
                e_curve = find_intersection(target, targetcirc_r, end, endcirc_r)

                try:
                    d_bet_se = sqrt(
                        (s_curve[0] - e_curve[0]) ** 2 + (s_curve[1] - e_curve[1]) ** 2
                    )

                    if (
                        abs((d_bet_se / 2) - targetcirc_r) < tolerance
                    ):  # In case 3 circles are on the same line.
                        dwg.add(
                            dwg.line(
                                start=s_curve,
                                end=e_curve,
                                stroke="blue",
                                stroke_width=1,
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
                    stroke="blue",
                    stroke_width=1,
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

                dwg.add(arc_path)
                # text = dwg.text(
                #     str(-sequence[(i + 3) % len(sequence)]),
                #     insert=start,
                #     fill="red",
                #     font_size="20px",
                #     text_anchor="middle",
                # )
                # dwg.add(text)

    dwg.save()
    print(f"SVG saved to {filename}")

# draws a planar multiloop from a permutation representation with given unbounded region using the circle packing algorithm
import svgwrite.path
from permrep import Multiloop
import matplotlib.pyplot as plt
import numpy as np
import svgwrite
from cmath import phase
from math import cos, sin
import os
import cmath


def generate_circles(
    multiloop: "Multiloop", inf_face: list[int]
) -> dict[int : tuple[complex, float]]:
    vertices_circ = {}
    edges_circ = {}
    faces_circ = {}

    curr_cir = 1
    for cycle in multiloop.sig.cycles:
        for he in cycle:
            vertices_circ[he] = curr_cir
        curr_cir += 1

    for cycle in multiloop.eps.cycles:
        edges_circ[cycle[0]] = curr_cir
        curr_cir += 1 if multiloop.is_samevert(cycle[0], cycle[1]) else 0
        edges_circ[cycle[1]] = curr_cir
        curr_cir += 1

    for cycle in multiloop.phi.cycles:
        for he in cycle:
            faces_circ[he] = curr_cir
        curr_cir += 1

    circ = {}
    for i in range(1, curr_cir):
        circ[i] = []

    def add_adj(circ, adj_circ):
        if adj_circ not in circ:
            circ.append(adj_circ)
        return circ

    for vert_he in vertices_circ:
        circ[vertices_circ[vert_he]] = add_adj(
            circ[vertices_circ[vert_he]], edges_circ[vert_he]
        )
        circ[vertices_circ[vert_he]] = add_adj(
            circ[vertices_circ[vert_he]], faces_circ[vert_he]
        )

    for edge_he in edges_circ:
        circ[edges_circ[edge_he]] = add_adj(
            circ[edges_circ[edge_he]], faces_circ[edge_he]
        )
        circ[edges_circ[edge_he]] = add_adj(
            circ[edges_circ[edge_he]], vertices_circ[edge_he]
        )

    for face_he in faces_circ:
        circ[faces_circ[face_he]] = add_adj(
            circ[faces_circ[face_he]], vertices_circ[face_he]
        )
        circ[faces_circ[face_he]] = add_adj(
            circ[faces_circ[face_he]], edges_circ[face_he]
        )
        circ[faces_circ[face_he]] = add_adj(
            circ[faces_circ[face_he]], edges_circ[-face_he]
        )

    external = {}
    for inf_he in inf_face:
        circ.pop(faces_circ[inf_he], None)
        circ.pop(edges_circ[inf_he], None)
        circ.pop(vertices_circ[inf_he], None)

        external[vertices_circ[inf_he]] = 1
        external[edges_circ[inf_he]] = 1
    internal = circ

    sequences = []

    for strand in multiloop.tau.cycles:
        this_seq = []
        for he in strand:
            this_seq.extend([edges_circ[he], he])
            if edges_circ[-he] not in this_seq:
                this_seq.extend([edges_circ[-he], -he])
            this_seq.extend([vertices_circ[he], he])

        sequences.append(this_seq)
    print(vertices_circ)
    print(edges_circ)
    print(faces_circ)
    return {"internal": internal, "external": external, "sequences": sequences}


def drawloop(
    circle_dict,
    filename="circle_pack.svg",
    scale=200,
    padding=50,
    sequence=None,
):
    if not isinstance(filename, (str, bytes, os.PathLike)):
        raise TypeError(f"Filename must be a string or path, got {type(filename)}")

    # Determine bounds
    min_x = min(z.real - r for z, r in circle_dict.values())
    max_x = max(z.real + r for z, r in circle_dict.values())
    min_y = min(z.imag - r for z, r in circle_dict.values())
    max_y = max(z.imag + r for z, r in circle_dict.values())

    width = (max_x - min_x) * scale + 2 * padding
    height = (max_y - min_y) * scale + 2 * padding

    dwg = svgwrite.Drawing(filename, size=(width, height))
    dwg.viewbox(0, 0, width, height)

    # Add white background
    dwg.add(dwg.rect(insert=(0, 0), size=(width, height), fill="white"))

    def to_svg_coords(center):
        cx = (center.real - min_x) * scale + padding
        cy = (max_y - center.imag) * scale + padding
        return (cx, cy)

    same_color = "#89CFF0"  # Light blue (can be changed)

    # Draw circles
    for name, (center, radius) in circle_dict.items():
        cx, cy = to_svg_coords(center)
        r = radius * scale
        dwg.add(
            dwg.circle(
                center=(cx, cy), r=r, fill=same_color, stroke="black", stroke_width=1
            )
        )
        dwg.add(
            dwg.circle(
                center=(cx, cy), r=2, fill="black", stroke="black", stroke_width=1
            )
        )
        dwg.add(
            dwg.text(
                str(name),
                insert=(cx + r * 0.1, cy),
                fill="black",
                font_size="15px",
                text_anchor="middle",
            )
        )

    # Draw connection lines (black, thick)
    if sequence and len(sequence) >= 2:
        offset_distance = (
            0.1  # Fraction of the distance between centers to offset the line
        )
        offset = 0
        for i in range(2, len(sequence), 4):
            a_center = circle_dict[sequence[i]][0]
            mid_center = circle_dict[sequence[(i + 2) % len(sequence)]][0]
            b_center = circle_dict[sequence[(i + 4) % len(sequence)]][
                0
            ]  # Wraparound to close loop

            start = to_svg_coords(a_center + offset)
            text = dwg.text(
                str(-sequence[(i + 3) % len(sequence)]),
                insert=start, 
                fill="red",
                font_size="20px",
                text_anchor="middle",
            )
            dwg.add(text)
            # Direction and offset
            vec = b_center - a_center
            distance = abs(vec)

            unit = vec / distance  # Unit vector in the direction of b -> a
            offset = unit * offset_distance * distance  # Offset from the centers
            # New start and end points with offset

            end = to_svg_coords(
                b_center + (offset if i + 2 < len(sequence) else 0)
            )  # Pull end point inward

            control = to_svg_coords(mid_center)

            # Draw line between offset points
            path = svgwrite.path.Path(
                d=f"M {start[0]},{start[1]} Q {control[0]},{control[1]} {end[0]},{end[1]}",
                fill="none",
                stroke="black",
                stroke_width=2,
            )

            # Add the path to your SVG drawing
            dwg.add(path)
            text = dwg.text(
                str(sequence[(i + 3) % len(sequence)]),
                insert=to_svg_coords(b_center - offset), 
                fill="red",
                font_size="20px",
                text_anchor="middle",
            )
            dwg.add(text)

    dwg.save()
    print(f"SVG saved to {filename}")

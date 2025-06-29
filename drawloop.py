# draws a planar multiloop from a permutation representation with given unbounded region using the circle packing algorithm
import svgwrite.path
from permrep import Multiloop
import matplotlib.pyplot as plt
import numpy as np
import svgwrite
from cmath import phase
from math import cos, sin, sqrt
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
        if multiloop.is_samevert(cycle[0], cycle[1]):
            curr_cir += 1
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
        # if vert_he not in inf_face:
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
        if edges_circ[edge_he] != edges_circ[-edge_he]:
            circ[edges_circ[edge_he]] = add_adj(
                circ[edges_circ[edge_he]], faces_circ[-edge_he]
            )
            circ[edges_circ[edge_he]] = add_adj(
                circ[edges_circ[edge_he]], edges_circ[-edge_he]
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
        circ.pop(edges_circ[-inf_he], None)
        circ.pop(vertices_circ[inf_he], None)

        external[vertices_circ[inf_he]] = 1
        external[edges_circ[inf_he]] = 1
        external[edges_circ[-inf_he]] = 1
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
    tolerance = 1e-9  # how accurately to approximate things
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
        for i in range(0, len(sequence), 2):
            startcirc_center = circle_dict[sequence[i]][0]
            startcirc_r = circle_dict[sequence[i]][1] * scale
            targetcirc_center = circle_dict[sequence[(i + 2) % len(sequence)]][0]
            targetcirc_r = circle_dict[sequence[(i + 2) % len(sequence)]][1] * scale
            endcirc_center = circle_dict[sequence[(i + 4) % len(sequence)]][0]
            endcirc_r = circle_dict[sequence[(i + 4) % len(sequence)]][1] * scale

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
                        stroke_width=2,
                    )
                )
                continue
            r_curve = sqrt(
                ((d_bet_se / 2) ** 2 * targetcirc_r**2)
                / (targetcirc_r**2 - (d_bet_se / 2) ** 2)
            )
            arc_path = svgwrite.path.Path(
                d=f"M {s_curve[0]},{s_curve[1]}",
                fill="none",
                stroke="blue",
                stroke_width=2,
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

            # Draw line between offset points
            # path = svgwrite.path.Path(
            #     d=f"M {start[0]},{start[1]} Q {control[0]},{control[1]} {end[0]},{end[1]}",
            #     fill="none",
            #     stroke="black",
            #     stroke_width=2,
            # )

            # # Add the path to your SVG drawing
            # dwg.add(path)
            # text = dwg.text(
            #     str(sequence[(i + 3) % len(sequence)]),
            #     insert=to_svg_coords(b_center - offset),
            #     fill="red",
            #     font_size="20px",
            #     text_anchor="middle",
            # )
            # dwg.add(text)
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

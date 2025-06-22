# draws a planar multiloop from a permutation representation with given unbounded region using the circle packing algorithm
from permrep import Multiloop
import matplotlib.pyplot as plt
import numpy as np


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
            this_seq.append(edges_circ[he])
            if edges_circ[-he] not in this_seq:
                this_seq.append(edges_circ[-he])
            this_seq.append(vertices_circ[he])
        this_seq.append(edges_circ[strand[0]])
            
        sequences.append(this_seq)
    return [internal, external, sequences]


def drawloop(circles, sequence):
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Colors for different circles
    colors = plt.cm.tab20(np.linspace(0, 1, len(circles)))

    # Plot each circle
    for i, (circle_id, (center, radius)) in enumerate(circles.items()):
        # Extract real and imaginary parts of center
        center_x = center.real
        center_y = center.imag

        # Create circle
        circle = plt.Circle(
            (center_x, center_y),
            radius,
            fill=False,
            color=colors[i],
            linewidth=2,
            label=f"Circle {circle_id}",
        )
        ax.add_patch(circle)

        # Add center point
        ax.plot(center_x, center_y, "o", color=colors[i], markersize=4)

        # Add circle ID label near the center
        ax.annotate(
            str(circle_id),
            (center_x, center_y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            color=colors[i],
            weight="bold",
        )

    # Set equal aspect ratio and grid
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Set axis limits with some padding
    all_centers = [center for center, radius in circles.values()]
    all_radii = [radius for center, radius in circles.values()]

    x_coords = [c.real for c in all_centers]
    y_coords = [c.imag for c in all_centers]
    max_radius = max(all_radii)

    x_min, x_max = min(x_coords) - max_radius, max(x_coords) + max_radius
    y_min, y_max = min(y_coords) - max_radius, max(y_coords) + max_radius

    # Add some padding
    padding = 0.1
    x_range = x_max - x_min
    y_range = y_max - y_min
    ax.set_xlim(x_min - padding * x_range, x_max + padding * x_range)
    ax.set_ylim(y_min - padding * y_range, y_max + padding * y_range)

    # Labels and title
    ax.set_xlabel("Real Part", fontsize=12)
    ax.set_ylabel("Imaginary Part", fontsize=12)
    ax.set_title(
        "Circle Visualization\n(Centers as complex numbers)", fontsize=14, weight="bold"
    )

    # Extract the centers for the sequence
    sequence_points = []
    for circle_id in sequence:
        center = circles[circle_id][0]  # Get center (first element of tuple)
        sequence_points.append((center.real, center.imag))

    # Plot the line segments
    for i in range(len(sequence_points) - 1):
        x1, y1 = sequence_points[i]
        x2, y2 = sequence_points[i + 1]
        ax.plot([x1, x2], [y1, y2], "k-", linewidth=1.5, alpha=0.7)

    # Add arrows to show direction
    for i in range(len(sequence_points) - 1):
        x1, y1 = sequence_points[i]
        x2, y2 = sequence_points[i + 1]
        # Add small arrow at midpoint
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        dx, dy = x2 - x1, y2 - y1
        # Normalize and scale arrow
        length = np.sqrt(dx**2 + dy**2)
        if length > 0:
            dx_norm, dy_norm = dx / length * 0.02, dy / length * 0.02
            ax.arrow(
                mid_x - dx_norm / 2,
                mid_y - dy_norm / 2,
                dx_norm,
                dy_norm,
                head_width=0.02,
                head_length=0.015,
                fc="red",
                ec="red",
                alpha=0.8,
            )

    # Add legend (but limit to avoid cluttering)
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) <= 10:
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    else:
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", ncol=2, fontsize=8)

    plt.tight_layout()
    plt.show()

    # Print summary information
    print("Circle Summary:")
    print("-" * 50)
    for circle_id, (center, radius) in sorted(circles.items()):
        print(
            f"Circle {circle_id:2d}: Center = ({center.real:6.3f}, {center.imag:6.3f}), Radius = {radius:.3f}"
        )

    print("\nConnection Sequence:")
    print("-" * 30)
    # print("Path:", " -> ".join(map(str, sequence)))

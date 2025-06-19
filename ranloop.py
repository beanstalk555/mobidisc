import random
from permrep import *


def generate(n: int) -> "Multiloop":
    # Generate a random multiloop with "n" vertices (4n half-edges)
    half_edges = [half_edge for i in range(1, 2 * n + 1) for half_edge in (i, -i)]
    random.shuffle(half_edges)
    return Multiloop([half_edges[i : i + 4] for i in range(0, len(half_edges), 4)])


def generate_planar(n: int) -> "Multiloop":
    # Generate a random planar multiloop with "n" vertices (4n half-edges)
    # TODO: Optimize this.
    generated_loop = generate(n)
    while not generated_loop.is_planar():
        generated_loop = generate(n)
    return generated_loop

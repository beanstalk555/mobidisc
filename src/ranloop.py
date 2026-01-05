import random
from permrep import *


def generate(n: int) -> "Multiloop":
    # Generate a random multiloop with "n" vertices (4n half-edges)
    half_edges = [half_edge for i in range(1, 2 * n + 1) for half_edge in (i, -i)]
    random.shuffle(half_edges)
    generated_loop = Multiloop([half_edges[i : i + 4] for i in range(0, len(half_edges), 4)])
    generated_loop.inf_face = generated_loop.phi[random.randint(0,len(generated_loop.phi)-1)]
    return generated_loop


def generate_planar(n: int) -> "Multiloop":
    # Generate a random planar multiloop with "n" vertices (4n half-edges)
    # TODO: Optimize this.
    MAX_ATTEMPTS = 100000
    for _ in range(MAX_ATTEMPTS):
        generated_loop = generate(n)
        if generated_loop.is_planar() and generated_loop.is_connected():
            return generated_loop
    raise RuntimeError("Too many failed attempts to find a planar loop.")

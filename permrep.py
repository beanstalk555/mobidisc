from typing import Iterable
from collections import deque


# data structure for storing a multiloop as a permutation representation
class Permutation:
    def __init__(self, cycles: list[list[int]], *, _generate_inv: bool = True) -> None:
        self.cycles = cycles
        self.perm = self.create_perm(cycles)

        # Automatically create the inverse permutation if requested, defaulted to True.
        if _generate_inv:
            self.inv = Permutation(
                [cycle[::-1] for cycle in cycles], _generate_inv=False
            )

    def __str__(self) -> str:
        return str(self.cycles)

    def __call__(self, half_edge: int) -> int:
        # Makes the object callable, equivalent to apply()
        return self.apply(half_edge)

    def __iter__(self) -> Iterable[int]:
        # Allows iteration over the half-edges in the permutation
        return iter(self.perm)

    def __len__(self) -> int:
        # Returns the number of cycles
        return len(self.cycles)

    def __getitem__(self, key: int) -> list[int]:
        # Access the k-th cycle directly
        return self.cycles[key]

    def __mul__(self, other: "Permutation") -> "Permutation":
        # Defines permutation composition: a * b = do a first, then do b
        visited = set()
        res = []
        for half_edge in self:
            if half_edge in visited:
                continue
            this_res = []
            current = half_edge
            while current not in visited:
                this_res.append(current)
                visited.add(current)
                # Apply self, then other
                current = other(self(current))
            res.append(this_res)
        res = Permutation(res)
        return res

    def create_perm(self, cycles: list[list[int]]) -> dict:
        # Converts cycle notation into dictionary with "next" references
        perm = dict()
        for cycle in cycles:
            for i in range(len(cycle) - 1):
                perm[cycle[i]] = cycle[i + 1]
            else:
                perm[cycle[-1]] = cycle[0]
        return perm

    def apply(self, half_edge: int) -> int:
        res = self.perm[half_edge]
        return res


class Multiloop:
    def __init__(self, cycles: list[list[int]], inf_face: list[int] = None) -> None:
        # Vertices: σ
        self.sig = Permutation(cycles)
        # Edges: ε
        self.eps = Permutation(self.generate_epsilon(cycles))
        # Faces: (σ ∘ ε)⁻¹
        self.phi = self.generate_area()
        # Strands: σ² ∘ ε
        self.tau = self.generate_strands()
        # Euler Characteristic V - E + F
        self.chi = self.cal_eulerchar()
        #
        self.inf_face = self.phi[0] if inf_face == None else inf_face

    def __str__(self) -> str:
        return f"Vertices: {self.sig}\nEdges: {self.eps}\nFaces: {self.phi}\nInfinite Face: {self.inf_face}\nStrands: {self.tau}\nEuler Characteristic: {self.chi}\nIs Planar: {self.is_planar()}\nIs Connected: {self.is_connected()}"

    def generate_epsilon(self, cycles: list[list[int]]) -> list[list[int]]:
        # Build ε as a set of +h and -h
        eps_cycles = []
        for cycle in cycles:
            for half_edge in cycle:
                if half_edge > 0:
                    eps_cycles.append([half_edge, -half_edge])
        return eps_cycles

    def generate_area(self) -> "Permutation":
        # φ = ε ∘ σ⁻¹
        areas = self.eps * self.sig.inv
        return areas

    def generate_strands(self) -> "Permutation":
        # τ = σ² ∘ ε
        strands = (self.sig * self.sig) * self.eps
        return strands

    def cal_eulerchar(self) -> int:
        # χ = V - E + F
        return len(self.sig) - len(self.eps) + len(self.phi)

    def is_planar(self) -> bool:
        # σ defines a planar (spherical) multiloop iff number of cycles in φ = n + 2, n = number of vertices
        return len(self.phi) == len(self.sig) + 2

    def is_connected(self) -> bool:
        all_halfedges = set(self.sig.perm.keys())
        start_halfedge = next(iter(all_halfedges))

        q = deque()
        visited = set()

        q.append(start_halfedge)

        while q:
            curr = q.popleft()
            if curr in visited:
                continue
            visited.add(curr)

            s = self.sig(curr)
            if s not in visited:
                q.append(s)

            e = self.eps(curr)
            if e not in visited:
                q.append(e)
        return visited == all_halfedges

    def is_samevert(self, edge1, edge2) -> bool:
        for _ in range(4):
            if self.sig(edge1) == edge2:
                return True
            edge1 = self.sig(edge1)
        return False
# data structure for storing a multiloop as a permutation representation
class Permutation:
    def __init__(self, cycles):
        self.perm = self.create_perm(cycles)

    def create_perm(self, cycles):
        vertex_perm = dict()
        for cycle in cycles:
            prev = cycle[-1]
            for i in range(len(cycle) - 1):
                vertex_perm[cycle[i]] = {"prev": prev, "next": cycle[i + 1]}
                prev = cycle[i]
            else:
                vertex_perm[cycle[-1]] = {"prev": prev, "next": cycle[0]}
        return vertex_perm

    def apply(self, half_edge):
        return self.perm[half_edge]["next"]

    def inv(self, half_edge):
        return self.perm[half_edge]["prev"]

    def __iter__(self):
        return iter(self.perm)

    def __str__(self):
        return str(self.perm)


class Multiloop:
    def __init__(self, sig: Permutation, eps: Permutation):
        self.sig = sig
        self.eps = eps
        self.areas = self.generate_area()

    def generate_area(self):
        visited = set()
        areas = []
        for half_edge in self.sig:
            if half_edge in visited:
                continue
            this_area = []
            current = half_edge
            while current not in visited:
                this_area.append(current)
                visited.add(current)
                current = self.sig.inv(self.eps.apply(current))
            areas.append(this_area)
        return areas

    def __str__(self):
        return f"Vertices:\n{self.sig}\nAreas:\n{self.areas}"

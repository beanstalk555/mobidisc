# data structure for storing a multiloop as a permutation representation
class Multiloop:
    def __init__(self, sig_raw=None):
        if sig_raw is None:
            sig_raw = []
        self.sig = self.create_vertex_perm(sig_raw)
        self.eps = self.create_edge_perm(sig_raw)
        self.areas = self.generate_area()

    def create_vertex_perm(self, loops):
        vertex_perm = dict()
        for loop in loops:
            prev = loop[-1]
            for i in range(len(loop) - 1):
                vertex_perm[loop[i]] = {"prev": prev, "next": loop[i + 1]}
                prev = loop[i]
            else:
                vertex_perm[loop[-1]] = {"prev": prev, "next": loop[0]}
        return vertex_perm

    def create_edge_perm(self, loops):
        edge_perm = dict()
        for loop in loops:
            for half_edge in loop:
                if half_edge > 0:
                    edge_perm[half_edge] = -half_edge
                    edge_perm[-half_edge] = half_edge
        return edge_perm

    def sig_func(self, half_edge):
        return self.sig[half_edge]["next"]

    def sig_inv_func(self, half_edge):
        return self.sig[half_edge]["prev"]

    def eps_func(self, half_edge):
        return self.eps[half_edge]

    def siginv_eps_func(self, half_edge):
        return self.sig_inv_func(-half_edge)

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
                current = self.siginv_eps_func(current)
            areas.append(this_area)
        return areas

    def __str__(self):
        return str(self.areas)

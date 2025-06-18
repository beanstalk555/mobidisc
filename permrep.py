# data structure for storing a multiloop as a permutation representation
class Permutation:
    def __init__(self, cycles, *, _generate_inv=True):
        self.cycles = cycles
        self.len = len(cycles)
        self.perm = self.create_perm(cycles)
        if _generate_inv:
            self.inv = Permutation(
                [cycle[::-1] for cycle in cycles], _generate_inv=False
            )

    def __call__(self, half_edge):
        return self.apply(half_edge)

    def __iter__(self):
        return iter(self.perm)

    def __str__(self):
        return str(self.cycles)

    def __len__(self):
        return self.len
    
    def __getitem__(self, key):
        return self.cycles[key]

    def __mul__(self, other):
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
                current = other(self(current))
            res.append(this_res)
        res = Permutation(res)
        return res

    def create_perm(self, cycles):
        vertex_perm = dict()
        for cycle in cycles:
            for i in range(len(cycle) - 1):
                vertex_perm[cycle[i]] = {"next": cycle[i + 1]}
            else:
                vertex_perm[cycle[-1]] = {"next": cycle[0]}
        return vertex_perm

    def apply(self, half_edge):
        res = self.perm[half_edge]["next"]
        return res


class Multiloop:
    def __init__(self, cycles, inf_face=None):
        # Vertices
        self.sig = Permutation(cycles)
        # Edges
        self.eps = Permutation(self.generate_epsilon(cycles))
        # Faces
        areas = self.generate_area()
        self.phi = areas
        # Strands
        self.tau = self.generate_strands()
        # nEuler Characteristic
        self.chi = self.cal_chi()

        # TODO: work on this
        self.inf_face = self.phi[0] if inf_face == None else inf_face

    def __str__(self):
        return f"Vertices:\n{self.sig}\nEdges:\n{self.eps}\nFaces:\n{self.phi}\nInfinite Face:\n{self.inf_face}\nStrands:\n{self.tau}\nEuler Characteristic:\n{self.chi}"

    def generate_epsilon(self, loops):
        edge_perm = []
        for loop in loops:
            for half_edge in loop:
                if half_edge > 0:
                    edge_perm.append([half_edge, -half_edge])
        return edge_perm

    def generate_area(self):
        areas = self.eps * self.sig.inv
        return areas

    def generate_strands(self):
        strands = (self.sig * self.sig) * self.eps
        return strands

    def cal_chi(self):
        return len(self.sig) - len(self.eps) + len(self.phi)

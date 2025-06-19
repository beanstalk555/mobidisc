# data structure for storing a multiloop as a permutation representation
class Permutation:
    def __init__(self, cycles, *, _generate_inv=True):
        self.cycles = cycles
        self.perm = self.create_perm(cycles)
        
        # Automatically create the inverse permutation if requested, defaulted to True.
        if _generate_inv:
            self.inv = Permutation(
                [cycle[::-1] for cycle in cycles], _generate_inv=False
            )

    def __str__(self):
        return str(self.cycles)

    def __call__(self, half_edge):
        # Makes the object callable, equivalent to apply()
        return self.apply(half_edge)

    def __iter__(self):
        # Allows iteration over the half-edges in the permutation
        return iter(self.perm)

    def __len__(self):
        # Returns the number of cycles
        return len(self.cycles)

    def __getitem__(self, key):
        # Access the k-th cycle directly
        return self.cycles[key]

    def __mul__(self, other):
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

    def create_perm(self, cycles):
        # Converts cycle notation into dictionary with "next" references
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
        # Vertices: σ
        self.sig = Permutation(cycles)
        # Edges: ε
        self.eps = Permutation(self.generate_epsilon(cycles))
        # Faces: (σ ∘ ε)⁻¹
        areas = self.generate_area()
        self.phi = areas
        # Strands: σ² ∘ ε
        self.tau = self.generate_strands()
        # Euler Characteristic V - E + F
        self.chi = self.cal_eulerchar()
        # 
        self.inf_face = self.phi[0] if inf_face == None else inf_face

    def __str__(self):
        return f"Vertices: {self.sig}\nEdges: {self.eps}\nFaces: {self.phi}\nInfinite Face: {self.inf_face}\nStrands: {self.tau}\nEuler Characteristic: {self.chi}\nIs Planar?: {self.is_planar()}"

    def generate_epsilon(self, loops):
        # Build ε as a set of involutive 2-cycles pairing +h and -h
        edge_perm = []
        for loop in loops:
            for half_edge in loop:
                if half_edge > 0:
                    edge_perm.append([half_edge, -half_edge])
        return edge_perm

    def generate_area(self):
        # φ = ε ∘ σ⁻¹
        areas = self.eps * self.sig.inv
        return areas

    def generate_strands(self):
        # τ = σ² ∘ ε
        strands = (self.sig * self.sig) * self.eps
        return strands

    def cal_eulerchar(self):
        # χ = V - E + F
        return len(self.sig) - len(self.eps) + len(self.phi)
    
    def is_planar(self):
        # σ defines a planar (spherical) multiloop iff number of cycles in φ = n + 2, n = number of vertices
        return len(self.phi) == len(self.sig) + 2

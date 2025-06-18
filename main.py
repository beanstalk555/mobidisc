import permrep as perm
def create_epsilon(loops):
    edge_perm = []
    for loop in loops:
        for half_edge in loop:
            if half_edge > 0:
                edge_perm.append([half_edge, -half_edge])
    return edge_perm

if __name__ == "__main__":
    loop = [[10,3,-1,-4],[6,1,-7,-2],[8,5,-9,-6],[2,7,-3,-8],[4,9,-5,-10]]
    example_sigma = perm.Permutation(loop)
    example_epsilon = perm.Permutation(create_epsilon(loop))
    example_loop = perm.Multiloop(example_sigma,example_epsilon)
    print(example_loop)
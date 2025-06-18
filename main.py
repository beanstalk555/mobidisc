import permrep as perm


def create_epsilon(loops):
    edge_perm = []
    for loop in loops:
        for half_edge in loop:
            if half_edge > 0:
                edge_perm.append([half_edge, -half_edge])
    return edge_perm


if __name__ == "__main__":
    loop = [
        [9, 2, -10, -3],
        [8, 3, -5, -4],
        [12, 7, -9, -8],
        [1, 10, -2, -11],
        [4, 5, -1, -6],
        [6, 11, -7, -12],
    ]
    example_loop = perm.Multiloop(loop)
    print(example_loop)

import permrep as perm

if __name__ == "__main__":
    #Vertex permutation
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

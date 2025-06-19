import permrep as perm
import ranloop

if __name__ == "__main__":
    #Vertex permutation
    multiloop = [
        [9, 2, -10, -3],
        [8, 3, -5, -4],
        [12, 7, -9, -8],
        [1, 10, -2, -11],
        [4, 5, -1, -6],
        [6, 11, -7, -12],
    ]
    random_multiloop = ranloop.generate_planar(10)
    example_loop = random_multiloop
    print(example_loop)
import permrep as perm
import ranloop
import drawloop
from circlepack import CirclePack
from collections import deque


if __name__ == "__main__":
    # Vertex permutation
    # multiloop = [[-1, -4, -3, -2], [1, 2, 6, 5], [3, 8, 7, -6], [-8, 4, -5, -7]]
    # example_loop = perm.Multiloop(multiloop)
    # example_loop.inf_face = [-1, 5, 4]
    multiloopEx = [
        (12,3,-9,-4),
        (1,10,-2,-11),
        (9,8,-10,-5),
        (4,5,-1,-6),
        (6,11,-7,-12),
        (7,2,-8,-3)
    ]
    #example_loop = perm.Multiloop(multiloopEx)
    #example_loop.inf_face =  (-1,6,10,-15,8)
    
    multiloopExFaces = [[]]
    
    def halfEdgePartnerSearch(multiloop: list[list[int]], halfEdge: int) -> tuple:
        
        for col in range(4):
            for row in range(len(multiloop)):
                if(multiloop[row][col] == halfEdge * -1):
                    return row, col
        raise LookupError("Could not find half edge")
    
    # only do for the first face base on start, implenet recurrsively later
    def faceGenerate(multiloop: list[list[int]], multiloopFaces: list[list[int]], startingHalfEdge= 1): 
        
        firstPartnerPos = halfEdgePartnerSearch(multiloop, startingHalfEdge)
        firstPartner = multiloop[firstPartnerPos[0]][firstPartnerPos[1]]
        
        partner =  firstPartnerPos
        nextHalfEdgeInCycle = 0
        while(nextHalfEdgeInCycle != startingHalfEdge):
            
            multiloopFaces[0].append(multiloop[partner[0]][partner[1]]) # as tuple
            
            if(firstPartner < 0): #continously to the left
                #multiloopRev = multiloop.copy()
                #multiloopRev.reverse()
                
                try:
                    nextHalfEdgeInCycle = multiloop[partner[0]][partner[1] - 1]
                except IndexError:
                    nextHalfEdgeInCycle = multiloop[partner[0]][2]
                                 
            else: # to the right
                try:
                    nextHalfEdgeInCycle = multiloop[partner[0]][partner[1] + 1]
                except IndexError:
                    nextHalfEdgeInCycle = multiloop[partner[0]][0]
            
            partner = halfEdgePartnerSearch(multiloop, nextHalfEdgeInCycle)
            
            
    faceGenerate(multiloopEx, multiloopExFaces, -4)
    print(multiloopExFaces)
        
    #cycle is really column in list of lists (permutations)
    def cycleContainsHalfEdge(multiloop: list[list[int]], cycleRow: int, halfEdge: int) -> bool:
        return multiloop[cycleRow].index(halfEdge) != None
    
    #searchMultiloop(multiloopEx, nextHalfEdgeInCycle)
                

    def faceGenerate(multiloop):
        crossSearched = deque([])
        crossSearched.append(multiloop[0][0])
        
    #faceGenerate(multiloopEx)
    

    # # TODO: Fix the problem where some randomly generated loop doesn't work with the circle packing algorithm
    # print(example_loop)
    # loop_to_circles = drawloop.generate_circles(example_loop, example_loop.inf_face)
    # print(
    #     f"{loop_to_circles["internal"]}\n{loop_to_circles["external"]}\n{loop_to_circles["sequences"]}"
    # )
    # # drawloop.drawloop(
    # #     CirclePack(loop_to_circles[0], loop_to_circles[1]),
    # #     loop_to_circles[2][0],
    # # )
    # # print(CirclePack(loop_to_circles[0], loop_to_circles[1]), loop_to_circles[2][0])
    # drawloop.drawloop(
    #     CirclePack(loop_to_circles["internal"], loop_to_circles["external"]),
    #     filename="circle_pack.svg",
    #     sequence=loop_to_circles["sequences"][0],
    # )

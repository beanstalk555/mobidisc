import permrep as perm
import ranloop
import drawloop
from circlepack import CirclePack
from collections import deque
import math


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

    ALL_LABELS = int(len(multiloopEx) * 4 / 2) # know 4 halfedges in cycle
    multilooplen = len(multiloopEx)
    GROUPS_OF_THREE = multilooplen + math.floor(multilooplen / 3)
    # amount of faces that MUST exist given the number of cycles

    multiloopExEdges = [(-num, num) for num in range(1, ALL_LABELS + 1)]
    
    multiloopExFaces = [[]] 
    for _ in range(1, GROUPS_OF_THREE):
        multiloopExFaces.append([])
    
    edgesAlreadyInLoop = []
    edgeQueue = deque([])
    
    
    def halfEdgeSearch(multiloop: list[list[int]], halfEdge: int, partner= True) -> tuple:
        
        for col in range(4):
            for row in range(len(multiloop)):
                if(partner and multiloop[row][col] == halfEdge * -1):
                    return row, col
                elif(not partner and multiloop[row][col] == halfEdge):
                    return row, col
        raise LookupError("Could not find half edge")
    
    def getNextHe(multiloop: list[list[int]], cycleCord: tuple) -> int:
        # always go clockwise back to original point
            try:
                return multiloop[cycleCord[0]][cycleCord[1] + 1]
            except IndexError:
                #invVertex = multiloop[partner[0]][::-1] # same as multiloop.reverse()
                #nextHalfEdgeInCycle = invVertex[3]
                return multiloop[cycleCord[0]][0]    
      
            
    
    
    # TODO: full recurrsive implementation
    
    # returns true if successful
    def faceGenerate(multiloop: list[list[int]], multiloopFaces: list[list[int]], startingHalfEdge, col) -> bool:
        
        if( edgesAlreadyInLoop.count(-1 * startingHalfEdge) > 0 ):
            return False

        firstPartnerPos = halfEdgeSearch(multiloop, startingHalfEdge)
        firstPartner = multiloop[firstPartnerPos[0]][firstPartnerPos[1]]
        
        partner =  firstPartnerPos
        nextHalfEdgeInCycle = 0
        
        while(nextHalfEdgeInCycle != startingHalfEdge):
            edge = multiloop[partner[0]][partner[1]]
            multiloopFaces[col].append(edge)
            edgesAlreadyInLoop.append(edge)
        
            nextHalfEdgeInCycle = getNextHe(multiloop, partner)
            
            # definitely way to so this with one less call
            edgeAcrossFromOGPartner = getNextHe( multiloop, halfEdgeSearch(multiloop, nextHalfEdgeInCycle, False) )

            # get starting edge and also all next edges (outside of the loop)
            if( not (edgesAlreadyInLoop.count(edgeAcrossFromOGPartner) > 0) ):
                edgeQueue.append(edgeAcrossFromOGPartner)
                
            partner = halfEdgeSearch(multiloop, nextHalfEdgeInCycle)
                
            
        # rework later so function returns a list that can be appended all at once in new column in multiloop   

        return True
        
    
    edgeQueue.append(1)
    multiloopCol = 0 
    
    while( bool(edgeQueue) ):
        print(edgeQueue)
        if( faceGenerate(multiloopEx, multiloopExFaces, edgeQueue.popleft(), multiloopCol)):
            multiloopCol = multiloopCol + 1
       
        
        
        
        
    print(multiloopExFaces)
    
    # multiloopExFaces2 = [[]]
    # faceGenerate(multiloopEx, multiloopExFaces2, -11, 0)
    # print("\n", multiloopExFaces2)
    
    
    # faceID = 0
    # for col in range(4):
    #     for row in range(len(multiloopEx)):
    #         multiloopExFaces.append([])
    #         faceGenerate(multiloopEx, multiloopExFaces, faceID, multiloopEx[row][col])
    #         faceID = faceID + 1
            
        
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

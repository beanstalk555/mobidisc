import permrep as perm
import ranloop
import drawloop
from circlepack import CirclePack
import math
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
    
    multilooplen = len(multiloopEx)
    multiloopExFaces = [[] ]
    #for i in range(multilooplen + math.floor(multilooplen / 3))
    
    loopsStartedFromHe = [] # list of halfedges started from to form a loop
    
    def halfEdgeSearch(multiloop: list[list[int]], halfEdge: int, findPartner: bool= True) -> tuple:
        
        for col in range(4):
            for row in range(len(multiloop)):
                if(findPartner and multiloop[row][col] == halfEdge * -1):
                    return row, col
                elif(multiloop[row][col] == halfEdge):
                    return row, col
        raise LookupError("Could not find half edge")
    
    def getHalfEdgeInCycle(multiloop: list[list[int]], firstPartner: int, currEdgeInCycle: tuple) -> int:
        if(firstPartner < 0): #continously to the left         
            try:
                return multiloop[currEdgeInCycle[0]][currEdgeInCycle[1] - 1]
            except IndexError:
                return multiloop[currEdgeInCycle[0]][2]                     
        else: # to the right
            try:
                return multiloop[currEdgeInCycle[0]][currEdgeInCycle[1] + 1]
            except IndexError:
                return multiloop[currEdgeInCycle[0]][0]
    
    # only do for the first face base on start, implenet recurrsively later
    def faceGenerate(multiloop: list[list[int]], multiloopFaces: list[list[int]], startingHalfEdge= 1, faceCol= 0): 
        
        loopsStartedFromHe.append(startingHalfEdge) # where we create face from
        firstPartnerPos = halfEdgeSearch(multiloop, startingHalfEdge)
        firstPartner = multiloop[firstPartnerPos[0]][firstPartnerPos[1]]
        
        partnerPos = firstPartnerPos
        print(multiloopExFaces)
        
        while (True): # do while loop
            multiloopFaces[faceCol].append(multiloop[partnerPos[0]][partnerPos[1]]) # as tuple
            
            nextHalfEdge = getHalfEdgeInCycle(multiloop, firstPartner, partnerPos)             
                     
            partnerPos = halfEdgeSearch(multiloop, nextHalfEdge)         
            # could optimize?
            nextHalfEdgePos = halfEdgeSearch(multiloop, nextHalfEdge, False) 
            nextnextHalfEdge = getHalfEdgeInCycle(multiloop, firstPartner, nextHalfEdgePos)
            
            if( not any(item == nextnextHalfEdge for item in loopsStartedFromHe) ):
                
                print(alreadyVisited)
                faceGenerate(multiloopEx, multiloopExFaces, nextnextHalfEdge, faceCol + 1)
                
            
            if(nextHalfEdge != startingHalfEdge):
                continue
            else:
                break
        
        
        
            
    faceGenerate(multiloopEx, multiloopExFaces, -10)
    
    
    print(multiloopExFaces)
    print(alreadyVisited)
        
        
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

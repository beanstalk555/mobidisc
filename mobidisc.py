import drawloop
import ranloop
import permrep as perm
from permrep import Multiloop
from drawloop import CircleAssignments
import numpy as np
import math
import vector
import queue

# treat top row as  start
def build_matrix(loop: Multiloop, sequences: list[list[int]], assignments: CircleAssignments, circle_dict: dict[any, tuple]) -> np.ndarray:
    
    # also need to catch out of bounds error or move to next sequence
    # catch later
    #while(not points_already_covered.empty() or sequences[curr_sequence]["circle_ids"].contains() ):
    
    n = len(loop.sig)  # num of verts per row
    q = np.array([[0 for _ in range(n)] for _ in range(n)], int)
        
    curr_seq = 0
    ci = list(sequences[curr_seq]["circle_ids"])
    ci_fil = [] 
    
    inv_vert_map = {val: key for key, val in assignments.vertices.items()}
    # filter to get only vert sequence
    for i in range(len(ci)):
        if(inv_vert_map.get(ci[i]) != None):
            ci_fil.append(ci[i])
            
    # quick way of removing duplicates
    cl_fil_nodupes = list(set(ci_fil))
    print(cl_fil_nodupes)
            
    print(ci)
    print(ci_fil)
                
    for _ in range(n):
        for i in range(1, n):
            q[i % n][ (i+1) % n ] = 1
            

            leftpt = circle_dict.get(ci_fil[(i - 1) % n])[0]
            midpt = circle_dict.get(ci_fil[(i) % n])[0]
            rightpt = circle_dict.get(ci_fil[(i + 1) % n])[0]
            
            leftpt = drawloop.to_svg_coords(leftpt)
            midpt = drawloop.to_svg_coords(midpt)
            rightpt = drawloop.to_svg_coords(rightpt)
            
            # if contain smonogon should always fail, implement later 
            if point_maintains_convex(leftpt, midpt, rightpt):
                q[i % n][ (i+2) % n ] = 1  
            else:
                q[i % n][ (i+2) % n ] = 0
            
    '''
    for i in range(n):
        for j in range(3, n):
            
            #first cond, Qij = Qkj = 1
            k = (j + i) / 2
            k = int(k - (k % 1))
            
            if(not (q[i][k] == 1 and q[k][j] == 1)):
                continue
            
            
            # 2nd cond counterclockwise
            dict_item1 = circle_dict.get(ci[i % n])
            dict_item2 = circle_dict.get(ci[j % n])
            dict_item3 = circle_dict.get(ci[k % n])
            
            
            point1 = (dict_item1[0].real, dict_item1[1].real)
            point2 = (dict_item2[0].real, dict_item2[1].real)
            point3 = (dict_item3[0].real, dict_item3[1].real)
            
            if(not (point_maintains_convex(point1, point2, point3))):
                continue
            
            # 3rd cond
            dict_item4 = circle_dict.get(ci[k + 1])
            dict_item5 = circle_dict.get(ci[k - 1])
            
            
            
        
            q[i][j] = 1
            
            return q
                    
            print(f"ij: {i, j} and k: {k}")
     
'''       
    return q
            

def is_cc():
    pass
            
# b being mid
def point_maintains_convex(a: tuple, b: tuple, c: tuple) -> bool:
    vA = np.array([a[0] - b[0], a[1] - b[1]], float) # b->a
    vB = np.array([c[0] - b[0], c[1] - b[1]], float) # b->c
    
    print(a[0])
    print(b[0])
    #print(vA)
    
    #vA_mag = math.sqrt(math.pow(vA[0], 2) * math.pow(vA[1], 2))
    #vB_mag = math.sqrt(math.pow(vB[0], 2) * math.pow(vB[1], 2))
    
    #vA_n = np.array([vA[0] / vA_mag, vA[1] / vA_mag], float)
    #vB_n = np.array([vB[0] / vB_mag, vB[1] / vB_mag], float)
    
    #dotProd = vA[0]*vB[0] + vA[1]*vB[1]
    #math.acos(dotProd / (vA_mag * vB_mag)) not for all quadrants 
    #math.degrees(math.acos(vA_n[0]))

    # rearranged law of cosines and dot product to get inner product space
    
    vA_u = vA / np.linalg.norm(vA)
    vB_u = vB / np.linalg.norm(vB)
    angleBetween = math.degrees(np.arccos(np.clip(np.dot(vA_u, vB_u), -1.0, 1.0)))
    
    #cross_prod = vA_mag * vB_mag * math.sin(angleBetween) * np.array[0, 1, 0]
    #cross_prod = np.linalg.cross(vA, vB)
    #print(angleBetween)

    if(angleBetween > 180):
        return True
    else:
        return False

'''
build_matrix(loop_to_circles["sequences"], loop_to_circles["assignments"], packed_circles)   
print_matrix(q) 
'''
'''
Created on 2017/09/01

@author: samejima
'''
import numpy as np
import sequences

def quantize(xyz, pixel_x = 0.0005, pixel_y = 0.0005):
    
    '''
    This calculates a normal map, nx, ny, nx over all pixels, 
    from xyz data that has continuous values of x, y, z, nx, ny, nz
    
    Parameters for quantization
    - pixel_x, pixel_y: size of each pixel
      (Large size leads to low resolution)
    - min_x, min_y: offset of x and y
    '''
    
    min_x, min_y = np.amin(xyz[:,0:2], axis =0)
    
    xyz[:, 0] =  np.array((xyz[:,0] - min_x )/pixel_x, dtype=int)
    xyz[:, 1] =  np.array((xyz[:,1] - min_y )/pixel_y, dtype=int)
    
    # If multiple values given in the same position, the average values are employed.   
    #      xy: position in pixels, index: index of position that has multiple values,
    #      count: the number of the multiple values at each position
    xy, index, count = np.unique(xyz[:,0:2], axis = 0,return_inverse=True,return_counts=True)
    quantized_xyz = np.column_stack(( xy, 
                                      np.bincount(index,xyz[:,2])/count,
                                      np.bincount(index,xyz[:,3])/count, 
                                      np.bincount(index,xyz[:,4])/count,
                                      np.bincount(index,xyz[:,5])/count ))
            
    return quantized_xyz

def generateLight(n_lights):
    
    gen = sequences.generate_hammersley(n_dims=3, n_points=n_lights)
    points = []
    for g in gen:
        points.append(g)
        
    # This returns points [0,1]^3
    L =  np.array(points)
    
    # The range [0,1] is scaled to [-1, 1]
    L = L * 2 - 1
    
    return L
   
def calculateAngle(v1,v2):
    v1 = v1/ np.linalg.norm(v1)
    v2 = v2/ np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1,v2),-1.0, 1.0))

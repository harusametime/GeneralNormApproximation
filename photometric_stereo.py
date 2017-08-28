'''
Created on 2017/08/25

@author: samejima
'''

import numpy as np
import scipy.sparse as sp
from general_norm import GeneralNorm
import sequences 

def quantize(xyz):
    
    '''
    This calculate a normal map that has nx, ny, nx over all pixels
    from xyz data that has continuous values of x, y, z, nx, ny, nz
    
    Parameters for quantization
    - pixel_x, pixel_y: size of each pixel
      (Large size leads to low resolution)
    - min_x, min_y: offset of x and y
    '''
    pixel_x = 0.0005;
    pixel_y = 0.0005;
    
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
   
if __name__ == '__main__':
    
    n_lights = 40;
    
    # xyz has the position x, y, z with normals nx, ny, nz in this order
    fname = "./data/bunny.xyz"
    xyz = np.loadtxt(fname)
    
    # q_xyx is a matrix of x, y, z, nx, ny, nz over all pixels
    q_xyz = quantize(xyz)
    
    # N as a matrix of Nx, Ny, Nz is extracted
    N = np.matrix(q_xyz[:, 3:6])
    
    # The light direction (uniformly distributed) is generated
    L = np.matrix(generateLight(n_lights))
    
    # Measurement
    M =  N * L.T
    
'''
Created on 2017/08/25

@author: samejima
'''

import numpy as np
import scipy.sparse as sp
from general_norm import GeneralNorm




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
    
    print xyz

if __name__ == '__main__':
    
    n_lights = 40;
    
    # xyz has the position x, y, z with normals nx, ny, nz in this order
    fname = "./data/bunny.xyz"
    xyz = np.loadtxt(fname)
    
    # N is a matrix of nx, ny, nz over all pixels
    N = quantize(xyz)
    
    # The light direction (uniformly distributed) is generated
    L = generateLight(n_lights)
    
'''
Created on 2017/08/31

@author: samejima
'''

import numpy as np
import scipy.sparse as sp
from scipy.linalg import toeplitz
from general_norm import GeneralNorm
import util
import sys

def readfile(source_type, path):
    
    # Only point cloud (poc) is acceptable.
    assert source_type == "poc", "Only point cloud is acceptable"
    
    # xyz has the position x, y, z with normals nx, ny, nz in this order
    xyz = np.loadtxt(path)
    
    # q_xyx is a matrix of x, y, z, nx, ny, nz over all pixels
    q_xyz = util.quantize(xyz)
    
    # N as a matrix of Nx, Ny, Nz is extracted
    N = np.matrix(q_xyz[:, 3:6])
    
    
    # Noise (not Gaussian) makes the measurement (noise) times larger (no noise if zero)
    noise = 5
    noise_ratio = 0.2
     
    # Add noise on N
    n_noise = int(N.size * noise_ratio)
    noise_index = np.array([np.random.choice(np.arange(N.shape[0]),n_noise)
                    ,np.random.choice(np.arange(N.shape[1]), n_noise)])
 
    N[noise_index[0], noise_index[1]] = noise * N[noise_index[0], noise_index[1]]
     
    g = np.concatenate((np.array(N[:,0]/N[:,2]).reshape((-1,)), np.array(N[:,1]/N[:,2]).reshape((-1,))))
    
    z = q_xyz[:,2]
    
    return z, g



        
if __name__ == '__main__':
    
    #Data source has to be poc (point cloud given as xyz)
    source_type = "poc"

    # File path for xyz 
    path = "./data/bunny.xyz"
    
    z, g = readfile(source_type, path)
    
    print toeplitz([1, 0, 0, 0], [1, 2, 3, 0, 0, 0])
    
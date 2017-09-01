'''
Created on 2017/08/31

@author: samejima
'''

import numpy as np
import scipy.sparse as sp
from scipy.sparse import spdiags
from scipy.linalg import block_diag
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

    xyz = q_xyz[:,0:3]
    
    return xyz, g
        
if __name__ == '__main__':
    
    #Data source has to be poc (point cloud given as xyz)
    source_type = "poc"

    # File path for xyz 
    path = "./data/bunny.xyz"
    
    xyz, g = readfile(source_type, path)
    pixels_x = int(np.max(xyz[:,0])) + 1
    pixels_y = int(np.max(xyz[:,1])) + 1
    n_pixels = pixels_x * pixels_y
    
    under_diag = np.zeros(n_pixels)
    for y in range(pixels_y):
        under_diag[pixels_x * y - 2] = 1
    bidiag_val = np.vstack((under_diag, np.ones(n_pixels), -np.ones(n_pixels)))
    D_x = spdiags(bidiag_val, np.array([-1,0,1]), n_pixels, n_pixels)
    
    under_diag = np.zeros(n_pixels)
    for x in range(pixels_x):
        under_diag[(pixels_y-2) * pixels_x + x] = 1
    bidiag_val = np.vstack((under_diag, -np.ones(n_pixels), np.ones(n_pixels)))
    D_y = spdiags(bidiag_val, np.array([-pixels_x,0,pixels_x]), n_pixels, n_pixels)
    
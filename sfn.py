'''
Created on 2017/08/31

@author: samejima
'''

import numpy as np
import scipy.sparse as sp
from scipy.sparse import spdiags
from scipy.linalg import block_diag
from scipy.sparse import vstack
from general_norm import GeneralNorm
import util
import sys
from numpy import int

def readfile(source_type, path):
    
    # Only point cloud (poc) is acceptable.
    assert source_type == "poc", "Only point cloud is acceptable"
    
    # xyz has the position x, y, z with normals nx, ny, nz in this order
    xyz = np.loadtxt(path)
    
    # q_xyx is a matrix of x, y, z, nx, ny, nz over all pixels
    q_xyz = util.quantize(xyz)
      
    return q_xyz

def addnoise(N):
    '''
    This adds noise to normal N. 
    The size and frequency of the noise are defined by "noise" and "noise ratio"
    '''
    
    # Noise (not Gaussian) makes the measurement (noise) times larger (no noise if zero)
    noise = 5
    noise_ratio = 0.2
     
    # Add noise on N
    n_noise = int(N.size * noise_ratio)
    noise_index = np.array([np.random.choice(np.arange(N.shape[0]),n_noise)
                    ,np.random.choice(np.arange(N.shape[1]), n_noise)])
 
    N[noise_index[0], noise_index[1]] = noise * N[noise_index[0], noise_index[1]]
    
    return N
    
        
if __name__ == '__main__':
    
    #Data source has to be poc (point cloud given as xyz)
    source_type = "poc"

    # File path for xyz 
    path = "./data/bunny.xyz"
    
    # pixel_size for quantization
    pixel_size = 0.0005
        
    # Formulation of a surface-from-normal problem (L1, L2)
    formulation ="L1"
    
    xyz = readfile(source_type, path)
    pixels_x = int(np.max(xyz[:,0])) + 1
    pixels_y = int(np.max(xyz[:,1])) + 1
    n_pixels = pixels_x * pixels_y
    
    # D_x is a differentiation matrix nx / nz
    # This is a tri-diagonal matrix of under_diag, 1, -1
    # under_diag is a vector that has 1 at pixels_x * y - 2 in matrix B.
    under_diag = np.zeros(n_pixels)
    for y in range(pixels_y):
        under_diag[pixels_x * y - 2] = 1
    bidiag_val = np.vstack((under_diag, np.ones(n_pixels), -np.ones(n_pixels)))
    D_x = spdiags(bidiag_val, np.array([-1,0,1]), n_pixels, n_pixels)

    # D_y is a differentiation matrix nx / nz
    # This is a tri-diagonal matrix of under_diag, 1, -1
    # under_diag has a vector of diagonal elements in identity matrix at last x_pixels.
    under_diag = np.zeros(n_pixels)
    for x in range(pixels_x):
        under_diag[(pixels_y-2) * pixels_x + x] = 1
    bidiag_val = np.vstack((under_diag, -np.ones(n_pixels), np.ones(n_pixels)))
    D_y = spdiags(bidiag_val, np.array([-pixels_x,0,pixels_x]), n_pixels, n_pixels)
    
    # D is a design matrix in this problem ||Dz -g||_p^p -> min.
    D = sp.vstack((D_x, D_y))
    
    # g_x,  is a vector of gradients calculated from normal
    g_x = np.zeros(n_pixels)
    g_y = np.zeros(n_pixels)
    
    # N as a matrix of Nx, Ny, Nz is extracted. Noise is added on them.
    N = np.array(xyz[:, 3:6])
    N_noise = addnoise(N)
    
    index_x = np.array(xyz[:,0], dtype=int)
    index_y = np.array(xyz[:,1], dtype=int)
    index = index_x + index_y * pixels_x

    g_x[index] = N[:,0]/N[:,2]
    g_y[index] = N[:,1]/N[:,2]
    
    # Gradient vector is a vector where g_x, g_y are aligned vertically.
    g = np.concatenate((g_x, g_y))
    
    # True depth values "true_z" from xyz
    true_z = np.zeros(n_pixels)
    z = np.zeros(n_pixels)
    
    true_z[index] = xyz[:,2]

    if formulation == "L2":
        list_A = [D]
        list_b = [g]
        w = np.array([1])
        l = np.array([2])
        p = GeneralNorm(list_A, list_b, w, l)
        z = p.solve() 
            
    elif formulation =="L1":
        list_A = [D]
        list_b = [g]
        w = np.array([1])
        l = np.array([1])
        p = GeneralNorm(list_A, list_b, w, l)
        z = p.solve()
        
    # Depth estimate z has constant ambiguity C.
    # C is adjusted by average of gaps between z - z_true, which
    # means that the surface estimate is translated to the true one.
    z = z * pixel_size
    gap = true_z - z
    C = np.sum(true_z[index] - z[index])/ index.shape[0]
    z= z - C
    
    print "L2 depth error :",
    print np.linalg.norm(z[index]-true_z[index], ord =2)
    
            
    
    
    
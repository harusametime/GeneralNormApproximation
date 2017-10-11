'''
Created on 2017/08/25

@author: samejima
'''

import numpy as np
import scipy.sparse as sp
from scipy.sparse import block_diag, kron, identity
import matplotlib.pyplot as plt
from general_norm import GeneralNorm
import cv2
import glob
import util
import time
import sys

def readfile(source_type, path, n_lights):
    # xyz has the position x, y, z with normals nx, ny, nz in this order
    if source_type == "poc":
        
        xyz = np.loadtxt(path)
        
        # q_xyx is a matrix of x, y, z, nx, ny, nz over all pixels
        q_xyz = util.quantize(xyz)
        
        # N as a matrix of Nx, Ny, Nz is extracted
        N = np.matrix(q_xyz[:, 3:6])
        
        # The light direction (uniformly distributed) is generated
        L = np.matrix(util.generateLight(n_lights))
        
        # Measurement
        M =  np.asarray(N * L.T)
        
    
        # Noise (not Gaussian) makes the measurement (noise) times larger (no noise if zero)
        noise = 5
        noise_ratio = 0.1
        
        # Add noise on N
        n_noise = int(M.size * noise_ratio)
        noise_index = np.array([np.random.choice(np.arange(M.shape[0]),n_noise)
                        ,np.random.choice(np.arange(M.shape[1]), n_noise)])
       
        M[noise_index[0], noise_index[1]] = noise * M[noise_index[0], noise_index[1]]
        
        
    elif source_type == "image":
        
        dirname = path
        L = np.loadtxt(dirname + "lighting.txt").T
        
        '''
        ground.txt is ground truth of normal map that was extracted 
        from mat file (matlab). If the image size is m * n, the file has
        m * 3n matrix, which horizontally aligns m * n matrices of nx, ny and nz. 
        '''
        N = np.loadtxt(dirname + "ground.txt")        
        pixels = int(N.shape[0]), int(N.shape[1]/3)
        
        N = np.vstack([N[:, int(pixels[1]*i): int(pixels[1]*(i+1))].reshape((-1,)) for i in range(3)]).T

        imgfiles = glob.glob(dirname + "*.*.png")
        # Read img file as a gray-scale image
        M = np.vstack([cv2.imread(imgfiles[i], flags =0).reshape((-1,)) for i in range(len(imgfiles))]).T
        
        # read mask file
        mask= cv2.imread(dirname +"mask.png", flags = 0)
        
        mask_flat = mask.reshape((-1,))
        mask_index = np.where(mask_flat == 255)
        
        # Extract only pixels that are defined in the mask (color is 255 in mask.png)
        N = N[mask_index]
        M = M[mask_index]

    return N, L, M, mask

if __name__ == '__main__':
    
    #Data source is poc (point cloud given as xyz) or image (given as PNG)
    #source_type = "poc"
    source_type = "image"
    
    # The number of light direction
    n_lights = 40;
    
    # File path for xyz or directory path for image
    #path = "./data/bunny.xyz"
    path = "./data/caesar/"
    
    # Read normal map and light direction from file
    N, L, M, mask = readfile(source_type, path, n_lights)
    
    # Formulation of a photometric-stereo problem(L1, L2)
    formulation ="L2"
    
    estimate_N = np.empty(N.shape)
    
    list_A = [sp.csr_matrix(L)]
    
    # Parameter for optimizer like lsqr, cg.
    opt_param = {'atol' : 1e-08, 'btol':1e-08, 'conlim':1e8, 'iter_lim':100000}
    
    
    '''
    Here the photometric stereo problem with formulation of L1, L2 is solved at each pixel i.
    '''
    if formulation == "L2":
        w = np.array([1])
        l = np.array([2])
        b = np.concatenate([M[i].T for i in range(M.shape[0])])
        list_b = [b]
        L = kron(identity(M.shape[0]), L)
        list_A = [L]
        p = GeneralNorm(list_A, list_b, w, l)
        estimate_N = p.solve(opt_param=opt_param)
        estimate_N = estimate_N.reshape(N.shape)

    elif formulation =="L1":
        w = np.array([1])
        l = np.array([1])
        b = np.concatenate([M[i].T for i in range(M.shape[0])])
        list_b = [b]
        L = kron(identity(M.shape[0]), L)
        list_A = [L]
        p = GeneralNorm(list_A, list_b, w, l)
        estimate_N = p.solve(optimizer="lsqr",opt_param=opt_param)
        estimate_N = estimate_N.reshape(N.shape)
        
    '''
    Evaluate angular error
    '''
    error = 0
    N = np.asarray(N)
    for i in range(M.shape[0]):
        error += util.calculateAngle(estimate_N[i], N[i])
    print("angular error (rad.): ",error/M.shape[0])
    
    '''
    Output normal map (converted to RGB values) 
    '''
    RGB = np.zeros((mask.shape[0] * mask.shape[1], 3))    
    RGB[np.where(mask == 255), :] = (N - np.amin(N, axis = 0)) * (255 / (np.amax(N, axis = 0) - np.amin(N, axis = 0)))
    print(np.where(mask == 255))
    print(RGB[0,:])
    RGB = RGB.astype(int).reshape((mask.shape[0], mask.shape[1], 3), order ='F')
    print(mask[0])
    print(RGB[0,0,:])
    cv2.imshow("test", RGB)
           
    

    
    
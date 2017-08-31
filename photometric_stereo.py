'''
Created on 2017/08/25

@author: samejima
'''

import numpy as np
import scipy.sparse as sp
from general_norm import GeneralNorm
import sequences
import cv2
import os
import sys
import glob

def quantize(xyz):
    
    '''
    This calculates a normal map, nx, ny, nx over all pixels, 
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
   
def calculateAngle(v1,v2):
    v1 = v1/ np.linalg.norm(v1)
    v2 = v2/ np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1,v2),-1.0, 1.0))

def readfile(source_type, path, n_lights):
    # xyz has the position x, y, z with normals nx, ny, nz in this order
    if source_type == "poc":
        
        xyz = np.loadtxt(path)
        
        # q_xyx is a matrix of x, y, z, nx, ny, nz over all pixels
        q_xyz = quantize(xyz)
        
        # N as a matrix of Nx, Ny, Nz is extracted
        N = np.matrix(q_xyz[:, 3:6])
        
        # The light direction (uniformly distributed) is generated
        L = np.matrix(generateLight(n_lights))
        
        # Measurement
        M =  np.asarray(N * L.T)
        
    elif source_type == "image":
        
        dirname = path
        L = np.loadtxt(dirname + "lighting.txt").T
        
        '''
        ground.txt is ground truth of normal map that was extracted 
        from mat file (matlab). If the image size is m * n, the file has
        m * 3n matrix, which horizontally aligns m * n matrices of nx, ny and nz. 
        '''
        N = np.loadtxt(dirname + "ground.txt")        
        pixels = N.shape[0], N.shape[1]/3
        
        N = np.vstack([N[:, pixels[1]*i : pixels[1]*(i+1)].reshape((-1,)) for i in range(3)]).T
        
        imgfiles = glob.glob(dirname + "*.*.png")
        # Read img file as a gray-scale image
        M = np.vstack([cv2.imread(imgfiles[i], flags =0).reshape((-1,)) for i in range(len(imgfiles))]).T
        
        # read mask file
        mask= cv2.imread(dirname +"mask.png", flags = 0)
        mask = mask.reshape((-1,))
        mask_index = np.where(mask == 255)

        # Extract only pixels that are defined in the mask (color is 255 in mask.png)
        N = N[mask_index]
        M = M[mask_index]

    return N, L, M

if __name__ == '__main__':
    
    #Data source is poc (point cloud given as xyz) or image (given as PNG)
    source_type = "image"
    
    # The number of light direction
    n_lights = 40;
    
    # File path for xyz or directory path for image
    #path = "./data/bunny.xyz"
    path = "./data/caesar/"
    
    # Read normal map and light direction from file
    N, L, M = readfile(source_type, path, n_lights)
    
    # Formulation of a photometric-stereo problem(L1, L2)
    formulation ="L1"
    
    
    # Noise (not Gaussian) makes the measurement (noise) times larger (no noise if zero)
    noise = 5
    noise_ratio = 0.1
    
    # Add noise on N
    noise_index = np.random.choice(np.arange(M.shape[0]), int(M.shape[0] * noise_ratio))
    M[noise_index] *= noise
    
    estimate_N = np.empty(N.shape)
    
    list_A = [sp.csr_matrix(L)]
    

    '''
    Here the photometric stereo problem with formulation of L1, L2 is solved at each pixel i.
    '''
    if formulation == "L2":
        w = np.array([1])
        l = np.array([2])
        for i in range(M.shape[0]):
            list_b = [M[i].T]  
            p = GeneralNorm(list_A, list_b, w, l)
            estimate_N[i] = p.solve() 
            
    elif formulation =="L1":
        w = np.array([1])
        l = np.array([1])
        for i in range(M.shape[0]):
            list_b = [M[i].T]
            p = GeneralNorm(list_A, list_b, w, l)
            estimate_N[i] = p.solve() 
        
    '''
    Evaluate angular error
    '''
    error = 0
    N = np.asarray(N)
    for i in range(M.shape[0]):
        error += calculateAngle(estimate_N[i], N[i])
    
    print "angular error (rad.): ",
    print error/M.shape[0]
    
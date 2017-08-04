'''
Created on 2017/08/03

@author: samejima
'''

import numpy as np
import scipy.sparse as sp
from general_norm import GeneralNorm

if __name__ == '__main__':
    
    row = np.array([0, 0, 1, 2, 2, 2])
    col = np.array([0, 2, 2, 0, 1, 2])
    data = np.array([1, 2, 3, 4, 5, 6])
    A = sp.csr_matrix((data, (row, col)), shape=(6, 3)).toarray()
    
    b = np.array([3,3,3,4,4,4])
    l = np.array([2,2])
    w = np.array([1,1])
        
    m = GeneralNorm(A, b, w, l)
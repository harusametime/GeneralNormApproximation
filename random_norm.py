'''
Created on 2017/08/03

@author: samejima
'''

import numpy as np
import scipy.sparse as sp
from general_norm import GeneralNorm

if __name__ == '__main__':
    
    # Number of rows in each matrix A_1, A_2,...
    # The dimensions are identical with the number of matrices.
    # For example, if A_1 and A_2 are give, the dimensions must be two.  
    n_row = np.array([300, 400])
    
    # Number of columns in A, which is identical with the size of solution vector x
    n_col = 400

    # Sparse matrices A_1, A_2,...  are randomly generated.
    list_A = [sp.csr_matrix(np.random.rand(n_row[i],n_col)) for i in range(n_row.shape[0])]
    
    # Ground truth x_gt is randomly determined.
    x_gt = np.random.rand(n_col)
    
    # Vector b is determined by A*x_gt
    list_b = [list_A[i] * x_gt  for i in range(n_row.shape[0])]
    
    # Weights for norms
    w = np.array([1,1])
    
    # p values for norms, e.g. [2, 2] is for Ridge Regression, [2,1] is for Lasso 
    l = np.array([2,1])
    
    # Put all parameters on general norms
    m = GeneralNorm(list_A, list_b, w, l)
    
    x = m.solve()
    
    # Evaluate error between x and x_gt by L2-norm
    print "Error (L2-norm):", np.linalg.norm(x-x_gt,ord =2)
'''
Created on 2017/08/03

@author: samejima
'''

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spal
import scipy.linalg as al
from itertools import product

import sys

class GeneralNorm(object):
    '''
    This class
    - represents w_1|A_1x-b_1|_L1^L_1 + w_2|A_2x-b_2|_L2^L_2 .... +w_K|A_K x-b_K|_L_K^L_K
    - solves minimization of the above formula with respect to x
    '''
    def __init__(self, list_A, list_b, w, l):
        '''
        A \in R^{(m_1 +m_2 +...)  x n}:
           Sparse matrix composed of vertically aligned matrices A_1, A_2, ..., A_K 
            CSR:Compressed Sparse Rows is recommended.
        b \in R^{(m_1 +m_2 +...)}:
            Vector composed of vertically aligned vectors b_1, ... b_K
        w \in R^{K}:
            Vector of weights w_1, ... w_k
        l \in R^{K}:
            Vector of p values in p-norm
        '''

        ## A, b, l, w are given on every norms
        assert len(list_A) == len(list_b) == w.shape[0] == l.shape[0], \
            "Error: Numbers of A, b, l, and w are %d, %d, %d, %d which need to be identical." %( len(list_A), len(list_b), w.shape[0], l.shape[0])
       
        self.A = sp.vstack(list_A)
        self.b = np.hstack(list_b)
        self.w = w
        self.l = l
        self.n_row = np.array([list_A[i].shape[0] for i in range(len(list_A))])

    
    def solve(self, optimizer = "lsqr", tol = 1.0e-6, max_itr = 1000):
        
        x_old = np.zeros(self.A.shape[1])
        x = np.zeros(self.A.shape[1])
        
        W = sp.identity(self.A.shape[0])
        diff = 1 # must be larger than tol
        itr = 0
        
        # Array of powers for calculating residual
        # If l = [2, 1], powerVec = [2,2,2,...,1,1,1...1] \in R^{m_1+m_2}
        powerVec = np.hstack([np.ones(self.n_row[i]) * self.l[i] for i in range(self.l.shape[0])])
        
        
        while diff > tol or max_itr > itr: 
            
            WA = W * self.A
            Wb = W * self.b
            
            if optimizer == "lsqr":
                # See here for exploiting x_old in LSQR.
                # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.sparse.linalg.lsqr.html
                r = Wb - WA * x_old
                x = spal.lsqr(WA, r)[0]    # Solution is the first returned value from lsqr.
                    
            if optimizer == "cg":
                x = spal.cg(WA, Wb, x0 = x_old)
            
            
            # Update values for stopping criteria  
            diff = al.norm(x-x_old, ord = 2)
            itr += 1
            
            residualVec = self.A * x - self.b
            residual = np.sum(np.power(residualVec, powerVec))
            
            print W.shape
            
#             for k in product(self.n_row):
#                 if self.l[k] < 2:
#                     W_k = sqrt(self.l[k]* self.w[k])/ 
#                 else:
#                     pass
            #np.apply_along_axis(updateW, axis, arr)
            
            sys.exit()
            
        return x
            
    def updateW(self, residual):
        pass
        
    
        
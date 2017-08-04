'''
Created on 2017/08/03

@author: samejima
'''

import numpy as np
import scipy.sparse as sp
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
        print self.n_row
    
    def solve(self, optimizer = "lsqr", tol = 1.0e-6, max_itr = 1000):
        
        x_old = np.zeros(self.A.shape[1])
        x = np.zeros(self.A.shape[1])
        
        W = sp.identity(self.A.shape[0])
        diff = 1 # must be larger than tol
        itr = 0
        
        
        while diff > tol or max_itr > itr: 
            
            WA = W * self.A
            Wb = W * self.b
            
            if optimizer == "lsqr":
                r = Wb - WA * x_old
                x = x_old + sp.linalg.lsqr(WA, r)
                
            if optimizer == "cg":
                x = sp.linalg.cg(WA, Wb, x0 = x_old)
                 
            
            diff = np.linalg.norm(x-x_old, ord = 2)
            itr += 1
            
            residualVec = self.A * x - self.b
            powerVec = np.ones()
            
            res = sum 
            
            print diff
            
            
        
    
        
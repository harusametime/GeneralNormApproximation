'''
Created on 2017/08/03

@author: samejima
'''

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spal
import scipy.linalg as al
from itertools import product
import math

import sys

class GeneralNorm(object):
    '''
    This class
    - represents w_1|A_1x-b_1|_L1^L_1 + w_2|A_2x-b_2|_L2^L_2 .... +w_K|A_K x-b_K|_L_K^L_K
    - solves minimization of the above formula with respect to x
    '''
    def __init__(self, list_A, list_b, w, p):
        '''
        A \in R^{(m_1 +m_2 +...)  x n}:
           Sparse matrix composed of vertically aligned matrices A_1, A_2, ..., A_K 
            CSR:Compressed Sparse Rows is recommended.
        b \in R^{(m_1 +m_2 +...)}:
            Vector composed of vertically aligned vectors b_1, ... b_K
        w \in R^{K}:
            Vector of weights w_1, ... w_k
        p \in R^{K}:
            Vector of p values in p-norm
        '''

        ## A, b, l, w are given on every norms
        assert len(list_A) == len(list_b) == w.shape[0] == p.shape[0], \
            "Error: Numbers of A, b, w, and p are %d, %d, %d, %d which need to be identical." %( len(list_A), len(list_b), w.shape[0], p.shape[0])
       
        self.A = sp.vstack(list_A)
        self.b = np.hstack(list_b)
        self.w = w
        self.p = p
        self.n_row = np.array([list_A[i].shape[0] for i in range(len(list_A))]) 
    
    def solve(self, optimizer = "lsqr", tol = 1.0e-6, eps = 1.0e-8, max_itr = 1000):
        
        x_old = np.zeros(self.A.shape[1])
        x = np.zeros(self.A.shape[1])
        
        W = sp.identity(self.A.shape[0])
        diff = 1 # must be larger than tol
        itr = 0
        
        # Array of w for calculating residual and updating W
        # If w = [2, 1], wVec = [2,2,2,...,1,1,1...1] \in R^{m_1+m_2}
        wVec = np.hstack([np.ones(self.n_row[i]) * self.w[i] for i in range(self.w.shape[0])])
        
        # Array of p for calculating residual and updating W
        # If p = [2, 1], pVec = [2,2,2,...,1,1,1...1] \in R^{m_1+m_2}
        pVec = np.hstack([np.ones(self.n_row[i]) * self.p[i] for i in range(self.p.shape[0])])
        
        # _updateW is a function of updating one element in W
        # This prepares a vectorized function, update_allW, that can update all elements in W.  
        update_allW = np.vectorize(self._updateW, otypes=[np.float])
        
        while diff > tol and max_itr > itr: 
            
            WA = W * self.A
            Wb = W * self.b
            
            if optimizer == "lsqr":
                # See here for exploiting x_old in LSQR.
                # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.sparse.linalg.lsqr.html
                r = Wb - WA * x_old
                x = spal.lsqr(WA, r)[0] +x_old   # Solution is the first returned value from lsqr.
                    
            if optimizer == "cg":
                x = spal.cg(WA, Wb, x0 = x_old)
            
            resVec = self.A * x - self.b
            
            # Update weight matrix
            W = sp.diags(update_allW(resVec, wVec, pVec, eps))
            
            # Update values for stopping criteria  
            diff = al.norm(x-x_old, ord = 2)
            itr += 1
            
            # Update the previous solution
            x_old = x
            
        return x
            
    def _updateW(self, res, w, p, eps = 1.0e-8):
        if  p < 2:
            return math.sqrt(p*w)/ max(pow(abs(res), 1 - 0.5*p), eps)
        else:
            return math.sqrt(p*w)/ pow(abs(res), 0.5*p-1)
            
        
    
        
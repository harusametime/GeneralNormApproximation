'''
Created on 2017/08/03

@author: samejima
'''

import numpy as np
import scipy.sparse as sp

class GeneralNorm(object):
    '''
    This class
    - represents w_1|A_1x-b_1|_L1^L_1 + w_2|A_2x-b_2|_L2^L_2 .... +w_K|A_K x-b_K|_L_K^L_K
    - solves minimization of the above formula with respect to x
    '''
    def __init__(self, A, b, w, l):
        '''
        A \in R^{mK x n}:
           Sparse matrix composed of vertically aligned matrices A_1, A_2, ..., A_K \in R^{m x n}
            CSR:Compressed Sparse Rows is recommended.
        b \in R^{mK}:
            Vector composed of vertically aligned vectors b_1, ... b_K  \in R^{m}
        w \in R^{K}:
            Vector of weights w_1, ... w_k
        l \in R^{K}:
            Vector of p values in p-norm
        '''
       

        ## Check dimensions ##
        assert A.shape[0] == b.shape[0], "Error: Numbers of rows in A and b are %d and %d, which need to be identical." %(A.shape[0], b.shape[0])
        assert w.shape[0] == l.shape[0], "Error: Numbers of rows in w and l are %d and %d, which need to be identical." %(w.shape[0], l.shape[0])
        
        self.A = A
        self.b = b
        self.w = w
        self.l = l
    
    def solve(self, optimizer = "lsqr"):
        pass
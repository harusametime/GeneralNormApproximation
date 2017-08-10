Fast General Norm Approximation 
====

## Description

This is a Python implemention of "Fast General Norm Approximation via Iteratively
Reweighted Least Squares" presented in The 4th ACCV workshop on e-Heritage 2016.


## Installation

Put *general_norm.py* in your directory. This program requires 
- Python 2.7
- Numpy
- Scipy

# Usage

```python
from general_norm import GeneralNorm
m = GeneralNorm(list_A, list_b, w, p)
x = m.solve()
```

`list_A` is a list of numpy ndarray *A*, `list_b` is a list of numpy ndarray *b*, `w` is a numpy array of weights on norms, and `p` is a numpy array of p values in l_p norms. We can get *x* as solution of the minimization problem.




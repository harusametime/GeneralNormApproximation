Fast General Norm Approximation 
====

## Description

This is a Python implementation of a solver for the following minimization problem:

<p align="center">
<img src ="https://latex.codecogs.com/gif.latex?\min_x&space;\sum_{k=1}^{K}&space;\lambda_k&space;\left&space;\|&space;A_k&space;x&space;-b_k&space;\right&space;\|_{p_k}&space;^{p_k}">
</p>

based on my work on "Fast General Norm Approximation via Iteratively Reweighted Least Squares" presented in The 4th ACCV workshop on e-Heritage 2016.

## Installation

Put *general_norm.py* in your directory. This program requires 
- Python 3.5
- Numpy
- Scipy
- opencv (for reading images in photometric_stereo.py)

## Usage

```python
from general_norm import GeneralNorm
m = GeneralNorm(list_A, list_b, w, p)
x = m.solve()
```

`list_A` is a list of scipy sparse matrices *A*, `list_b` is a list of numpy ndarray *b*, `w` is a numpy array of weights on norms, and `p` is a numpy array of p values in l_p norms. We can get *x* as solution of the minimization problem.

## Example

### *random_norm.py* 

Design matrices *A* and solution *x* are randomly determined, and *b* is calculated by *A* *x*. Then the problem to minimize the sum of norms ||*A* *x* - *b*||_p^p is solved with respect to *x*.

### *photometric_stereo.py* 

Normal map *N* of an object is estimated from measurement *M* and light direction *L* based on Lambert's law. This program can receive either point cloud with normals or measurement images of an object under different light directions. 

### *sfn.py*
Surface represented by a set of depth is estimated from normal map *N* based on the fact that differentiation of the surface is corresponding to gradient, which can be calculated from normal. 

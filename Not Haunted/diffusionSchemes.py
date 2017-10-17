# Numerical schemes for simulating diffusion for outer code diffusion.py

from __future__ import absolute_import, division, print_function
import numpy as np

# The linear algebra package for BTCS (for solving the matrix equation)
import scipy.linalg as la

def FTCS(phiOld, d, nt):
    "Diffusion of profile in phiOld using FTCS using non-dimensional diffusion"
    "coefficient, d"
    
    nx = len(phiOld)
    
    # new time-step array for phi
    phi_new = phiOld.copy()
    
    # FTCS for all time steps
    for it in xrange(int(nt)):
        
        for count in xrange(1, nx-1):
            
            phi_new[count] = d*phiOld[count+1] + (1-2*d)*phiOld[count] + \
                             d*phiOld[count-1]
                             
        phi_new[0] = phi_new[1]
        phi_new[-1] = phi_new[-2]
        # zero gradient at the boundary, conditions
        phiOld = phi_new.copy()
        # So that the new phi old is the old phi new
        
    return phi_new
    

def BTCS(phi, d, nt):
    "Diffusion of profile in phi using BTCS using non-dimensional diffusion"
    "coefficient, d, assuming fixed value boundary conditions"
    
    nx = len(phi)
    
    # array representing BTCS
    M = np.zeros([nx, nx])
    #Zero gradient boundary conditions
    M[0, 0] = 1.
    M[0, 1] = -1.
    M[-1, -1] = 1.
    M[-1, -2] = -1.
    for i in xrange(1, nx-1):
        M[i, i-1] = -d
        M[i, i] = 1+2*d
        M[i, i+1] = -d
        
    # BTCS for all time steps
    for it in xrange(int(nt)):
        # RHS for zero gradient boundary conditions
        phi[0] = 0
        phi[-1] = 0
        
        phi = la.solve(M, phi)
        
    return phi
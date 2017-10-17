# Various function for plotting results and for calculating error measures

### Copy out most of this code. Code commented with 3#s (like this) ###
### is here to help you to learn python and need not be copied      ###

### If you are using Python 2.7 rather than Python 3, import various###
### functions from Python 3 such as to use real number division     ###
### rather than integer division. ie 3/2  = 1.5  rather than 3/2 = 1###
from __future__ import absolute_import, division, print_function

### The numpy package for numerical functions and pi                ###
import numpy as np

# Import the special package for the erf function
from scipy import special

def analyticErf(x, Kt, alpha, beta):
    "The analytic solution of the 1d diffusion equation with diffusion"
    "coeffienct K at time t assuming top-hat initial conditions which are"
    "one between alpha and beta and zero elsewhere"
    
    phi = 0.5 * special.erf((x-alpha)/np.sqrt(4*Kt))  \
        - 0.5 * special.erf((x-beta )/np.sqrt(4*Kt))
    return phi


def L2ErrorNorm(phi, phiExact):
    "Calculates the L2 error norm (RMS error) of phi in comparison to"
    "phiExact, ignoring the boundaries"
    
    #remove one of the end points
    phi = phi[1:-1]
    phiExact = phiExact[1:-1]
    
    # calculate the error and the error norms
    phiError = phi - phiExact
    L2 = np.sqrt(sum(phiError**2)/sum(phiExact**2))

    return L2
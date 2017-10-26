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
    "and whose boundary conditions are zero at both infinity and negative"
    "infinity"
    
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
    
def analyticAlt(x, Kt, inf):
    "Analytic solution of 1d diffusion equations for initial top hat between"
    "0.4 and 0.6, where inf is the number of terms taken in fourier series"
    "With the same boundary conditions of zero flux out of the ends as"
    "numerical schemes"
    
    phi = 0.2
    
    for m in xrange(1, inf):
        
        phi += -(2/(np.pi*m))*np.sin(4*m*np.pi/5)*np.cos(2*m*x*np.pi)*\
                                                np.exp(-4*np.pi**2*m**2*Kt)
        
    return phi
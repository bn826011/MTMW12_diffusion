import numpy as np

# Initial conditions function for diffusion

def squareWave(x,alpha,beta):
    "A square wave as a function of position, x, which is 1 between alpha"
    "and beta and zero elsewhere. The initialisation is conservative so"
    "that each phi contains the correct quantity integrated over a region"
    "a distance dx/2 either side of x"
    
    phi = np.zeros_like(x)
    
    # The grid spacing (assumed uniform)
    dx = x[1] - x[0]
    
    # Set phi away from the end points (assume zero at the end points)
    for j in xrange(1,len(x)-1):
        # edges of the grid box (using west and east notation)
        xw = x[j] - 0.5*dx
        xe = x[j] + 0.5*dx
        
        #integral quantity of phi
        phi[j] = max((min(beta, xe) - max(alpha, xw))/dx, 0)

    return phi
    
def worseWave(x, alpha, beta):
    "A non-conservative, naive initialisation of the above square wave"
    "simply samples function for initial conditions at grid point"
    "will have different area beneath depending on size of x"
    "Area tends to 0.2 from below as len(x) tends to infinity"

    phi = np.zeros_like(x)
    
    for j in xrange(1, len(x)-1):
        
        if (x[j] > alpha) & (x[j] < beta):
            
            phi[j] = 1
            
    return phi
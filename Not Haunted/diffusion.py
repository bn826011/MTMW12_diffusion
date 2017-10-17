#!/usr/bin/python

# Outer code for setting up the diffusion problem on a uniform
# grid and calling the function to perform the diffusion and plot.

from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt

# read in all the linear advection schemes, initial conditions and other
# code associated with this application
execfile("diffusionSchemes.py")
execfile("diagnostics.py")
execfile("initialConditions.py")

def main(xmin = 0., xmax = 1., nx = 21, nt = 20, dt = 0.1, K = 2e-3, 
         squareWaveMin = 0.4, squareWaveMax = 0.6):
    "Diffuse a square wave between squareWaveMin and squareWaveMax on a domain"
    "between x = xmin and x = xmax split over nx spatial steps with diffusion"
    "coefficient K, time step dt for nt time steps"
    
    # Derived parameters
    dx = (xmax - xmin)(nx -1)
    d = K*dt/dx**2  # Non-dimensional diffusion coefficient
    print("Non-dimensional diffusion coefficient = ", d)
    print("dx = ", dx, " dt = ", dt, " nt = ", nt)
    print("end time = ", nt*dt)
    
    # spatial points for plotting and for defining initial conditions
    x = np.zeros(nx)
    for j in xrange(nx):
        x[j] = xmin + j*dx
    print('x = ', x)
    
    # Initial conditions
    phiOld = squareWave(x, squareWaveMin, squareWaveMax)
    # Analytic solution (of square wave profile in an infinite domain)
    phiAnalytic = analyticErf(x, K*dt*nt, squareWaveMin, squareWaveMax)
    
    # Diffusion using FTCS and BTCS
    phiFTCS = FTCS(phiOld.copy(), d, nt)
    phiBTCS = BTCS(phiOld.copy(), d, nt)
    
    # Calculate and print out error norms
    print("FTCS L2 error norm = ", L2ErrorNorm(phiFTCS, phiAnalytic))
    print("BTCS L2 error norm = ", L2ErrorNorm(phiBTCS, phiAnalytic))
    
    # Plot the solutions
    font = {'size' : 20}
    plt.rc('font', **font)
    
    plt.figure(1)
    plt.clf()
    # clear figure
    plt.ion()
    # interactive on
    plt.plot(x, phiOld, label='Initial', color='black')
    plt.plot(x, phiAnalytic, label='Analytic', color='black', 
             linestyle = '--', linewidth=2)
    plt.plot(x, phiFTCS, label='FTCS', color='blue')
    plt.plot(x, phiBTCS, label='BTCS', color='red')
    plt.axhline(0, linestyle=':', color='black')
    plt.ylim([0, 1])
    plt.legend(bbox_to_anchor=(1.1, 1))
    plt.xlabel('$x$')
    plt.savefig('plots/FTCS_BTCS.pdf')
    
main()
    
    
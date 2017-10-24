#!/usr/bin/python

# Outer code for setting up the diffusion problem on a uniform
# grid and calling the function to perform the diffusion and plot.

# Also script style 'epilogue' for ease of reproduction of results
# with printed commentary

from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt

# read in all the linear advection schemes, initial conditions and other
# code associated with this application
execfile("diffusionSchemes.py")
execfile("diagnostics.py")
execfile("initialConditions.py")

def main(xmin = 0., xmax = 1., nx = 41, nt = 40, dt = 0.1, K = 1e-3, 
         squareWaveMin = 0.4, squareWaveMax = 0.6, stringmod = '', 
         printindex = 1):
    "Diffuse a square wave between squareWaveMin and squareWaveMax on a domain"
    "between x = xmin and x = xmax split over nx spatial steps with diffusion"
    "coefficient K, time step dt for nt time steps"
    "stringmod is an appendage to figure filenames to distinguis runs"
    "if plotindex = 1, it will print values and figures, if 0 it will not"
    
    # Derived parameters
    dx = (xmax - xmin)/(nx -1)
    d = K*dt/dx**2  # Non-dimensional diffusion coefficient
    
    if printindex == 1:
        print("Non-dimensional diffusion coefficient = ", d)
        print("dx = ", dx, " dt = ", dt, " nt = ", nt)
        print("end time = ", nt*dt)
    
    # spatial points for plotting and for defining initial conditions
    x = np.zeros(nx)
    for j in xrange(nx):
        x[j] = xmin + j*dx
        
    if printindex == 1:
        print('x = ', x)
    
    # Initial conditions
    phiOld = squareWave(x, squareWaveMin, squareWaveMax)
    # Analytic solution (of square wave profile in an infinite domain)
    phiAnalytic = analyticErf(x, K*dt*nt, squareWaveMin, squareWaveMax)
    
    # Diffusion using FTCS and BTCS
    phiFTCS = FTCS(phiOld.copy(), d, nt)
    phiBTCS = BTCS(phiOld.copy(), d, nt)
    
    # Calculate and print out error norms
    FTL2en = L2ErrorNorm(phiFTCS, phiAnalytic)
    BTL2en = L2ErrorNorm(phiBTCS, phiAnalytic)
    
    if printindex == 1:
    
        print("FTCS L2 error norm = ", FTL2en)
        print("BTCS L2 error norm = ", BTL2en)
    
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
        plt.ylabel('$\phi$')
        plt.savefig('plots/FTCS_BTCS' + stringmod + '.pdf')
        plt.show()
    
        plt.figure(2)
        plt.clf()
        plt.ion()
        plt.plot(x, phiFTCS-phiAnalytic, label='FTCSerror', color='blue')
        plt.plot(x, phiBTCS-phiAnalytic, label='BTCSerror', color='red')
        plt.legend(bbox_to_anchor=(1.1, 1))
        plt.xlabel('$x$')
        plt.ylabel('$\Delta \phi$')
        plt.savefig('plots/FTCS_BTCS_error' + stringmod + '.pdf')
        plt.show()
        
    return [FTL2en, BTL2en]

def stabilitytest(nxvec = np.arange(3, 101), nt = 40, dt = 0.1):
    "Function to produce log plot of error with increasing x increment"
    "to provide indication of stability"
    
    xno = np.size(nxvec)
    
    FTL2en = np.zeros(xno)
    BTL2en = np.zeros(xno)
    
    for counter in xrange(xno):
        
        [FTL2en[counter], BTL2en[counter]] = main(nx = nxvec[counter], 
                                                         printindex = 0)
    
    plt.figure(3)
    plt.clf()
    plt.semilogy(nxvec, FTL2en, label='FTCS', color='blue')
    plt.semilogy(nxvec, BTL2en, label='BTCS', color='red')
    plt.legend(bbox_to_anchor=(0.6, 1))
    plt.xlabel('$nx$')
    plt.ylabel('L2 error norm')
    plt.savefig('plots/FTCS_BTCS_errornorm' + stringmod + '.pdf')
    plt.show()

# #Question 1
#main()
# #Question 2
#print('These initial and boundary conditions represent a substance in a' +
#' pipe with closed ends, starting concentrated in the centre 20%.' +
#' Pipe volume: 1, Substance volume: 0.2.')
# #Question 3
#main(nt = 400, stringmod = '_nt400')
#main(nt = 1000, stringmod = '_nt1k')
#print('The numerical results are so far from the analytic solution as ' +
# 'they employ different boundary conditions. The numerical solution ' +
# 'requires zero spatial gradient at the endpoints representing a closed ' +
# 'pipe, wherease the boundary conditions of the analytic solutions are that ' +
# 'the concentrations at $\pm \infty$ are zero representing an infinitely ' +
# 'long pipe.')
#print('This results in the conservation of area beneath the line ' +
# 'beneath our numerical curves, settling out at an even distribution at ' +
# '0.2 as t $\ rightarrow$ $\infty$. On the other hand, we do not get ' +
# 'conservation of area in the range 0 - 1 for the analytic solution, as it ' +
# 'is a solution for an infinite domain. As t $\ rightarrow$ $\infty$ the ' +
# 'analytic solution will tend to zero everywhere.')
#  #Question 4
#stabilitytest()
#print('To test the stability of my implementations of FTCS and BTCS, my' +
#' experiment calculates the L2 error norm of the numerical solutions ' +
#'obtained using number of x steps ranging from 3 to 100 to represent ' +
#' the range 0-1, and 40 t steps of size 0.1s.')
#print('The results of this are presented in the graph, and we can see that ' +
#'the FTCS solution's L2 error norm seems to tend to infinity as x tends to ' +
#'infinity, growing almost exponentially, whereas the BTCS solution's L2 ' +
#'error norm seems to settle at around 10^-2 as x tends to infinity.')
#print('This would suggest that )
 
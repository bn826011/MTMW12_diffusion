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
         printindex = 1, ylimits = [0, 1], usebetteranalyticsolution = 0,
         inf = 100, useworseinitialconditions = 0):
    "Diffuse a square wave between squareWaveMin and squareWaveMax on a domain"
    "between x = xmin and x = xmax split over nx spatial steps with diffusion"
    "coefficient K, time step dt for nt time steps"
    "stringmod is an appendage to figure filenames to distinguis runs"
    "if plotindex = 1, it will print values and figures, if 0 it will not"
    "the better analytic solution is one with the same boudary conditions as"
    "the numerical scheme"
    
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
    
    if useworseinitialconditions == 1:
        phiOld = worseWave(x, squareWaveMin, squareWaveMax)
        
    # Analytic solution (of square wave profile in an infinite domain)
    phiAnalytic = analyticErf(x, K*dt*nt, squareWaveMin, squareWaveMax)
    
    if usebetteranalyticsolution == 1:
        phiAnalytic = analyticAlt(x, K*dt*nt, max(inf, nx))
    
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
        #font = {'size' : 20}
        #plt.rc('font', **font)
    
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
        plt.ylim(ylimits)
        plt.legend(bbox_to_anchor=(1, 1))
        plt.xlabel('$x$')
        plt.ylabel('$\phi$')
        plt.savefig('plots/FTCS_BTCS' + stringmod + '.pdf')
        plt.show()
    
        plt.figure(2)
        plt.clf()
        plt.ion()
        plt.plot(x, phiFTCS-phiAnalytic, label='FTCSerror', color='blue')
        plt.plot(x, phiBTCS-phiAnalytic, label='BTCSerror', color='red')
        plt.legend(bbox_to_anchor=(1, 1))
        plt.xlabel('$x$')
        plt.ylabel('$\Delta \phi$')
        plt.savefig('plots/FTCS_BTCS_error' + stringmod + '.pdf')
        plt.show()

    return [FTL2en, BTL2en]
    

def stabilitytestx(nxvec = np.arange(3, 101), nt = 40, dt = 0.1):
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
    plt.savefig('plots/FTCS_BTCS_errornorm.pdf')
    plt.show()
    
def stabilitytestt(nxvec = [21, 51, 101, 201], ntvec = 5*2**np.arange(10)):
    "Function to produce log plot of error with increasing t increment"
    "to provide indication of stability. Always runs for 2, 4 & 8 seconds."
    
    dtvec = 4/ntvec    
    
    print(dtvec)
    
    tno = np.size(ntvec)
    xno = np.size(nxvec)
    
    FTL2en = np.zeros([tno, xno, 3])
    BTL2en = np.zeros([tno, xno, 3])
    
    for cx in xrange(xno):
        
        for ct in xrange(tno):
        
            [FTL2en[ct, cx, 1], BTL2en[ct, cx, 1]] = main(nx = nxvec[cx], 
                               nt = ntvec[ct], dt = dtvec[ct], printindex = 0)
                               
            [FTL2en[ct, cx, 0], BTL2en[ct, cx, 0]] = main(nx = nxvec[cx], 
                              nt = ntvec[ct]/2, dt = dtvec[ct], printindex = 0)
                              
            [FTL2en[ct, cx, 2], BTL2en[ct, cx, 2]] = main(nx = nxvec[cx], 
                              nt = 2*ntvec[ct], dt = dtvec[ct], printindex = 0)
                              
    colour = np.array([[0.6, 0, 0], [1, 0.2, 0.2], [0.2, 0.2, 1], [0, 0, 0.5]])
    
    plt.figure(4)
    plt.clf()
    for cx in xrange(xno):
        
        plt.loglog(dtvec, FTL2en[:, cx, 0], #label='nx = '+str(nxvec[cx])+', t = 2s', 
                   color = colour[cx], linestyle = ':')
        plt.loglog(dtvec, FTL2en[:, cx, 1], #label='nx = '+str(nxvec[cx])+', t = 4s', 
                   color = colour[cx], linestyle = '--')
        plt.loglog(dtvec, FTL2en[:, cx, 2], label='nx = '+str(nxvec[cx]),#+', t = 8s', 
                   color = colour[cx], linestyle = '-')
    
    plt.legend(bbox_to_anchor=(0.35, 1))
    plt.ylim([1e-5, 1e40])
    plt.xlabel('$dt$')
    plt.ylabel('FTCS L2 error norm')
    plt.savefig('plots/FTCS_errornorm.pdf')
    plt.show()
    
    plt.figure(5)
    plt.clf()
    for cx in xrange(xno):
        
        plt.loglog(dtvec, BTL2en[:, cx, 0], #label='nx = '+str(nxvec[cx])+' t = 2s', 
                   color = colour[cx], linestyle = ':')
        plt.loglog(dtvec, BTL2en[:, cx, 1], #label='nx = '+str(nxvec[cx])+' t = 4s', 
                   color = colour[cx], linestyle = '--')
        plt.loglog(dtvec, BTL2en[:, cx, 2], label='nx = '+str(nxvec[cx]),#+' t = 8s', 
                   color = colour[cx], linestyle = '-')
    
    plt.legend(bbox_to_anchor=(1, 0.4))
    plt.xlabel('$dt$')
    plt.ylabel('BTCS L2 error norm')
    plt.savefig('plots/BTCS_errornorm.pdf')
    plt.show()

def ordertest(nxvec = 5*np.arange(1, 21)+1, d = 0.1, K = 1e-3, ylims = 1,
              stringmod = ''):
    "Function to test the order of the numerical schemes FTCS & BTCS"
    "nt = (4K/d)(nx-1)^2, and so nx must be chosen s.t. nt is an integer"
    
    dxvec = 1/(nxvec -1)
    dtvec = d/(K*(nxvec - 1)**2)
    ntvec = 4/dtvec
    
    xno = np.size(nxvec)
    
    FTL2en = np.zeros(xno)
    BTL2en = np.zeros(xno)
    
    for counter in xrange(xno):
        
        [FTL2en[counter], BTL2en[counter]] = main(nx = nxvec[counter], 
                      nt = ntvec[counter], dt = dtvec[counter], printindex = 0,
                      usebetteranalyticsolution = 1)
    
    orderFx = np.zeros(xno-1)
    orderFt = np.zeros(xno-1)
    orderBx = np.zeros(xno-1)
    orderBt = np.zeros(xno-1)
    
    for cn in xrange(xno-1):
        
        orderFx[cn] = (np.log(FTL2en[cn])-np.log(FTL2en[cn+1]))/\
                                        (np.log(dxvec[cn])-np.log(dxvec[cn+1]))                              
        orderFt[cn] = (np.log(FTL2en[cn])-np.log(FTL2en[cn+1]))/\
                                        (np.log(dtvec[cn])-np.log(dtvec[cn+1]))    
        orderBx[cn] = (np.log(BTL2en[cn])-np.log(BTL2en[cn+1]))/\
                                        (np.log(dxvec[cn])-np.log(dxvec[cn+1]))    
        orderBt[cn] = (np.log(BTL2en[cn])-np.log(BTL2en[cn+1]))/\
                                        (np.log(dtvec[cn])-np.log(dtvec[cn+1]))
                                        
    plt.figure(6)
    plt.clf()
    plt.semilogx(dtvec[0:-1], orderFx, color='blue', linestyle='-',
             label = 'FTCS error order in x')
    plt.semilogx(dtvec[0:-1], orderFt, color='blue', linestyle='--',
             label = 'FTCS error order in t')
    plt.semilogx(dtvec[0:-1], orderBx, color='red', linestyle='-',
             label = 'BTCS error order in x')
    plt.semilogx(dtvec[0:-1], orderBt, color='red', linestyle='--',
             label = 'BTCS error order in t')   
    if ylims == 1:
        plt.ylim([0, 2.5])
    plt.xlabel('$\Delta t = d(\Delta x)^2/K$')
    plt.ylabel('Calculated order of method')
    plt.legend(bbox_to_anchor=(1.3, 1.1))
    plt.savefig('plots/ordertest' + stringmod + '.pdf')
    plt.show()

    plt.figure(7)
    plt.clf()
    plt.loglog(dtvec, FTL2en, color='blue', label='FTCS')
    plt.loglog(dtvec, BTL2en, color='red', label='BTCS')
    plt.loglog(dtvec, 0.046*(dtvec), color='black', label='0.046*$\Delta t$')
    if ylims == 1:
        plt.clf()
        plt.xlim([0, 0.5])
        plt.ylim([0, 0.03])
        plt.plot(dtvec, FTL2en, color='blue', label='FTCS')
        plt.plot(dtvec, BTL2en, color='red', label='BTCS')
        plt.plot(dtvec, 0.046*(dtvec), color='black', label='0.046*$\Delta t$')        
    plt.xlabel('$\Delta t = d(\Delta x)^2/K$')
    plt.ylabel('L2 error norm')
    plt.legend(bbox_to_anchor=(0.4, 1))
    plt.savefig('plots/errorbydt' + stringmod + '.pdf')
    plt.show()
        
    plt.figure(8)
    plt.clf()
    plt.loglog(dxvec, FTL2en, color='blue', label='FTCS')
    plt.loglog(dxvec, BTL2en, color='red', label='BTCS')
    plt.loglog(dxvec, 4.6*(dxvec)**2, color='black', label='4.6*$\Delta x^2$')
    if ylims == 1:
        plt.clf()
        plt.xlim([0, 0.05])
        plt.ylim([0, 0.05])
        plt.plot(dxvec, FTL2en, color='blue', label='FTCS')
        plt.plot(dxvec, BTL2en, color='red', label='BTCS')
        plt.plot(dxvec, 4.6*(dxvec)**2, color='black', label='4.6*$\Delta x^2$')
    plt.xlabel('$\Delta x$')
    plt.ylabel('L2 error norm')
    plt.legend(bbox_to_anchor=(0.3, 1))
    plt.savefig('plots/errorbydx' + stringmod + '.pdf')
    plt.show()
    
    

#print('For some reason I cannot use LaTeX on the computors in the Met '+
#'department, so I am instead forced to type my write-up into Phython.')
#print('Apologies')
#print('Question 1')
#main()
#print('Question 2')
#print('These initial and boundary conditions represent a substance in a' +
#' pipe with closed ends, starting concentrated in the centre 20%.')
#print('Pipe volume: 1, Substance volume: 0.2.')
#print('Question 3')
#main(nt = 400, stringmod = '_nt400')
#main(nt = 1000, stringmod = '_nt1k')
#print('The numerical results are so far from the analytic solution as ' +
# 'they employ different boundary conditions. The numerical solution ' +
# 'requires zero spatial gradient at the endpoints representing a closed ' +
# 'pipe, wherease the boundary conditions of the analytic solutions are that '+
# 'the concentrations at $\pm \infty$ are zero representing an infinitely ' +
# 'long pipe.')
#print('This results in the conservation of area beneath the line ' +
# 'beneath our numerical curves, settling out at an even distribution at ' +
# '0.2 as t $\ rightarrow$ $\infty$. On the other hand, we do not get ' +
# 'conservation of area in the range 0 - 1 for the analytic solution, as it ' +
# 'is a solution for an infinite domain. As t $\ rightarrow$ $\infty$ the ' +
# 'analytic solution will tend to zero everywhere.')
#main(nt = 400, stringmod = '_nt400_bas', usebetteranalyticsolution = 1)
#print('We can see that by comparing to an analytic solution using the same '+
#'initial conditions as our numerical method the errors are greatly reduced '+
#'and area beneath the graph is, to a better approximation, conserved')
#print('Question 4')
#stabilitytestx()
#print('To test the stability of my implementations of FTCS and BTCS, my' +
#' experiment calculates the L2 error norm of the numerical solutions ' +
#'obtained using number of x steps ranging from 3 to 100 to represent ' +
#' the range 0-1, and 40 t steps of size 0.1s.')
#print('The results of this are presented in the graph, and we can see that ' +
#'the FTCS solution\'s L2 error norm seems to tend to infinity as x tends to '+
#'infinity, growing almost exponentially, whereas the BTCS solution\'s L2 ' +
#'error norm seems to settle at around 10^-2 as x tends to infinity.')
#print('This would suggest that FTCS is unstable for some combination of ' +
#'dx and dt, whereas BTCS is not.')
#stabilitytestt()
#print('To further test stabilty, and also to compare the behaviour of the ' +
#'schemes to theoretical prediction, I have run a similar experiment but ' +
#'varying dt. This experiment is plotted for four values of nx, and three '+
#'lengths of time, 2 (dotted line), 4 (dashed line) and 8 (solid line) '+
#'seconds. This again would indicate that BTCS is stable, the L2 error norm '+
#'decreasing with the longer run times. The errors in ' +
#'FTCS however (bearing in mind that this is an ' +
#'O(1) system, even errors of O(10) are huge), for small dt decrease ' +
#'with longer time runs, but above some critical value of dt depending on ' +
#'nx increase very rapidly, quite possibly to infinity, indicating again '+
#'that FTCS is unstable.')
#print('These critical values of dt are predicted analytically to be '+
#'1.25, 0.2, 0.05 and 0.0125, for nx = 21, 51, 101 and 201 respectively, '+
#'for FTCS. The plot for the FTCS L2 error norm indicates that the scheme '+
#'behaves as predicted by theory. Theory also predicts that BTCS is stable '+
#'for all time step lenghts dt, and the plot indicates that the scheme is '+
#'also in agreement with this.')
#main(nx = 51, nt = 40, dt = 0.1, stringmod = '_nx51nt40')
#main(nx = 51, nt = 16, dt = 0.25, stringmod = '_nx51nt16')
#main(nx = 51, nt = 16, dt = 0.25, ylimits = [-5, 5], stringmod = '_nx51nt16full')
#print('A curious thing is a lack of observable oscillations for 0.25 < d < 0.5')
#main(nx = 141, nt = 16, dt = 0.025, stringmod = '_nx141nt16')
#print('The reason for which perhaps being either that the analytic ' +
#'equation merely indicates that oscillations are possible rather than ' +
#'predicted, that the oscillations occur on a smaller wavelength than the '+
#'grid spacing used, and so are therefore not calculated, or that the ' +
#'amplitudes are so small in comparison to 1, or the background error, that '+
#'they are inpercievable in our plots. Whatever the reason, I have managed '+
#'find one instance illustrating oscillations for d<0.5 in nx = 141, dt = 0.025')
#main(nx = 36, nt = 40, dt = 0.4, stringmod = '_nx36')
#main(nx = 36, nt = 40, dt = 0.4, stringmod = '_nx36bas', usebetteranalyticsolution = 1)
#print('The oscillations are slightly clearer when comparing it to an '
#'alternative analytic solution with the same boundary conditions, but not '
#'by much. These enhanced oscillations may be due to either the reduction in '+
#'error in comparision to the error function solution, or due to the removal '+
#'of higher modes in the plotting of the fourier solution. By having as least '+
#'as many modes as there are x gridpoints I have hoped to remove the latter '+
#'as an option.')
#print('Question 5')
#print('From fairly thorough Taylor analysis we can see that the error in '+
#'our solution for phi $\epsilon \approx A\Delta t + B(\Delta x)^2$.')
#print('Through our error analysis we will hold $d = \frac{K\Delta t}{(\Delta x)^2}'+
#'to be constant. Therefore for this particular numerical scheme, conveniently, '+
#'we can substitute this in giving us that $\epsilon \approx (Ad/K + B)(\Delta x)^2 '+
#'\approx (A + BK/d)(\Delta t)$. This means that with two values of the error '+
#'$\epsilon_1$ and $\epsilon_2$, with corresponding $\Delta x_1, \Delta x_2, '+
#'\Delta t_1 and \Delta t_2$ we should see that \frac{log(\epsilon_1)-log(\epsilon_2)}'+
#'{log(\Delta x_1) - log(\Delta x_2)} = 2, and \frac{log(\epsilon_1)-log(\epsilon_2)}'+
#'{log(\Delta t_1) - log(\Delta t_2)} = 1, as theory would predict.')
#print('I feel the need to note that this is an artefact of both our '+
#'differential equation and our numerical method being first order in time '+
#'and second order in space, and would not work with, for example, the CTCS '+
#'for which the numerical method was second order in time, but the differential '+
#'equation only first order in time (also it is unstable everywhere).')
#ordertest()
#ordertest(ylims = 0, stringmod = '_full')
#print('')
#print('Question 6')
#main(stringmod = '_wic', useworseinitialconditions = 1)
#main(nt = 1000, stringmod = '_nt1k_wic', useworseinitialconditions = 1)
#main(nt = 1000, stringmod = '_nt1k_bas_wic', usebetteranalyticsolution = 1,
# useworseinitialconditions = 1)
#print('From all this we can see that the more naive initial conditions do '
#'not start off with the same area beneath the plot of 0.2 as the other '
#'initial conditions and our analytic solutions do. This causes the '
#'numerical solution to be less than the analytic solution.')
#main(nt = 5000, stringmod = '_nt5k')
#main(nt = 5000, stringmod = '_nt5kbas', usebetteranalyticsolution = 1)
#print('Although something interesting to obsterve is that the better initial '+
#'conditions at very long times have a total area just slightly larger than '+
#'that of the analytic solution')
#print('Question 7')

 

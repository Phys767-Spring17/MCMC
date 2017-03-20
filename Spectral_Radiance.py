from __future__ import print_function

import emcee
import corner
import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as pl
from matplotlib.ticker import MaxNLocator
from scipy.optimize import curve_fit
import PyAstronomy

h = 6.626070040*(10**(-34))  #J*s
c = 299792458 #m/s
k_b = 1.38064852*(10**(-23))  #m^2*kg*s^-2*K^-1

a = 2*h*(c**2)
b = (h*c)/k_b

def my_own_Planck(x,T):
    #x is the wavelength
    #returns spectral radiance per sr
    return((a/x**5)*(1/((np.exp(b/(T*x))-1))))

h = 6.626e-34
c = 3.0e+8
k = 1.38e-23

def planck(wav, T):
    a = 2.0*h*c**2
    b = h*c/(wav*k*T)
    intensity = a/ ( (wav**5) * (np.exp(b) - 1.0) )
    return intensity
"""
X = np.linspace(0,3,256,endpoint=True)

Y1_A = (planck(X*10**-6,3000))
Y2_A = (planck(X*10**-6,4000))
Y3_A = (planck(X*10**-6,5000))

plot1, = pl.plot(X,Y1_A, color="red", label='3000 K')
plot2, = pl.plot(X,Y2_A, color="green", label='4000 K')
plot3, = pl.plot(X,Y3_A, color="blue", label='5000 K')
pl.legend([plot1,plot2,plot3],['3000 K','4000 K','5000 K'])
pl.xlabel("Wavelength (um)")
pl.ylabel("Spectral Radiance")

pl.show()
"""
X = np.linspace(3,23,256,endpoint=True)

T1 = 25
T2 = 4000
T3 = 5000

Y1_B = (np.log10(planck(X*10**-6,T1)))
Y2_B = (np.log10(planck(X*10**-6,T2)))
Y3_B = (np.log10(planck(X*10**-6,T3)))

x = np.asarray([3.368,4.618,12.082,22.194,3.6,4.5])
y = np.asarray([-18.435,-17.042,-15.728,-16.307,-18.063,-17.173])
yerr = np.asarray([0.087,0.087,0.087,0.087,0.043,0.043])

pl.errorbar(x, y, yerr=yerr, fmt=".k")

#plot1B, = pl.plot(X,Y1_B, color="red", label='3000 K')

#plot2B, = pl.plot(X,Y2_B, color="green", label='4000 K')
#plot3B, = pl.plot(X,Y3_B, color="blue", label='5000 K')
#pl.legend([plot1B],['3000 K'])
pl.xlabel("Wavelength (um)")
pl.ylabel("LOG10(Spectral Radiance)")

pl.show()

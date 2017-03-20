#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import emcee
import corner
import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as pl
from matplotlib.ticker import MaxNLocator
np.seterr(divide='ignore', invalid='ignore')
np.seterr(over='ignore', invalid='ignore')

# Reproducible results!
np.random.seed(123)

# Choose the "true" parameters.
m_true = -0.9594
b_true = 4.294
f_true = 0.534


data = [[3.368,-18.435,0.087],[4.618,-17.042,0.087],[12.082,-15.728,0.087],[22.194,-16.307,0.087
],[3.6,-18.063,0.043],[4.5,-17.173,0.043]]

x = np.asarray([3.368,4.618,12.082,22.194,3.6,4.5])
y = np.asarray([-18.435,-17.042,-15.728,-16.307,-18.063,-17.173])
yerr = np.asarray([0.087,0.087,0.087,0.087,0.043,0.043])

h = 6.626070040*(10**(-34))  #J*s
c = 299792458 #m/s
k_b = 1.38064852*(10**(-23))  #m^2*kg*s^-2*K^-1

a = 2*h*(c**2)
b = (h*c)/k_b

# Plot the dataset and the true model.
xl = np.array([.0001, 10])
pl.errorbar(x, y, yerr=yerr, fmt=".k")
#pl.plot(xl, m_true*xl+b_true, "k", lw=3, alpha=0.6)
#pl.ylim(-9, 9)
pl.xlabel("$x$")
pl.ylabel("$y$")
pl.tight_layout()
pl.savefig("line-data.png")

# Do the least-squares fit and compute the uncertainties.
A = np.vstack((np.ones_like(x), x)).T
C = np.diag(yerr * yerr)
cov = np.linalg.inv(np.dot(A.T, np.linalg.solve(C, A)))
T_ls = np.dot(cov, np.dot(A.T, np.linalg.solve(C, y)))

print("""Least-squares results:
    T = {0} ± {1} (truth: {2})
""".format(T_ls[0], np.sqrt(cov[1, 1]), "IDONNOEITHER"))
# Plot the least-squares result.
#print((a/xl**5)*(1/((np.exp(b/(xl*T_ls))-1))),"this is it")
pl.plot(xl, (a/xl**5)*(1/((np.exp(b/(xl*T_ls[0]))-1))), "--k")
pl.savefig("line-least-squares.png")

# Define the probability function as likelihood * prior.
def lnprior(theta):
    print(theta,"SADASD")
    a, T, lnf = theta
    if 0.0 < T < 60000.0 and -10.0 < lnf < 1.0:
        return 0.0
    return -np.inf

def lnlike(theta, x, y, yerr):
    a, T, lnf = theta
    model = (a/x**5)*(1/((np.exp(b/(x*T))-1)))
    inv_sigma2 = 1.0/(yerr**2 + model**2*np.exp(2*lnf))
    return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))

def lnprob(theta, x, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)

# Find the maximum likelihood value.
chi2 = lambda *args: -2 * lnlike(*args)
result = op.minimize(chi2, [0,0,0], args=(x, y, yerr))
a, T_ml, lnf_ml = result["x"]
print("""Maximum likelihood result:
    T = {0} (truth: {1})
    f = {2} (truth: {3})
""".format(T_ml, "IDONNO", np.exp(lnf_ml), "IALSODONNO"))

# Plot the maximum likelihood result.
pl.plot(xl, (a/xl**5)*(1/((np.exp(b/(xl*T_ml))-1))), "k", lw=2)
pl.savefig("line-max-likelihood.png")

# Set up the sampler.
ndim, nwalkers = 3, 100
pos = [result["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr))

# Clear and run the production chain.
print("Running MCMC...")
sampler.run_mcmc(pos, 500, rstate0=np.random.get_state())
print("Done.")

# Plot the least-squares result.
pl.plot(sampler.chain[:, :, 0].T, color="k", alpha=0.4)
pl.savefig("line-time.png")

# Make the triangle plot.
burnin = 50
samples = sampler.chain[:,burnin:, :].reshape((-1, 1))

"""
fig = corner.corner(samples, labels=["$m$", "$b$", "$\ln\,f$"],
                      truths=[m_true, b_true, np.log(f_true)])
fig.savefig("line-triangle.png")
"""
# Plot some samples onto the data.
pl.figure()
for T_ls in samples[np.random.randint(len(samples), size=100)]:
    pl.plot(xl, (a/xl**5)*(1/((np.exp(b/(xl*T_ls))-1))), color="k", alpha=0.1)
#pl.plot(xl, m_true*xl+b_true, color="r", lw=2, alpha=0.8)
pl.errorbar(x, y, yerr=yerr, fmt=".k")
pl.xlabel("$x$")
pl.ylabel("$y$")
pl.tight_layout()
pl.savefig("line-mcmc.png")
# Compute the quantiles.
samples[:] = np.exp(samples[:])
T_mcmc = list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],
                                                axis=0))))
print("""MCMC result:
    T = {0[0]} +{0[1]} -{0[2]}
""".format(T_mcmc))

#! /usr/bin/env python
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This script implements the EM algorithm to fit a set of 2-dimensional Gaussian clusters
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from numpy import *
from scipy.cluster.vq import *
import matplotlib
from matplotlib.font_manager import FontProperties
matplotlib.use("Agg")
import pylab
from MVNClusters import *
from EM_GMM import *

# Expectation-maximization works much better when there is a good initial guess for the cluster centroids.

# But, this guess can come from a local maximum, in which case EM will get stuck and not 

# For now we sove these problems crudely, by calling k-means before E-M to get starting centroids, and repeating the process
# several times and using the fit parameters that return the maximum likelihood.  This is a rough way to assure that we're finding
# something close to the global maximum.


# Define a function to calculate the multivariate normal distribution likelihood 
# for a single point 'x' and a single MVN distribution defined by 'mu' and 'cov'.
#
# This function is used for plotting contours for the fitted MVN distributions, it's
# not used in the calculation of the parameters.
def MVNLikelihood(x,mu,cov):
    L = exp(-0.5*dot(x-mu,dot(linalg.inv(cov),x-mu)))
    return L


# Define some data to fit.

k = 6
dim = 2

# The MVNClusters function will return a set of 'k' randomly-oriented MVN clusters in 'dim'-dimensional space
# Hasn't been tested on dim>2.

clean_obs,x,y,cls_labels = MVNClusters(k,400,400)

# Generate some background noise across the range of parameter space covered by the observations returned by MVNClusters
#
# Currently this algorithm doesn't handle noise very well, it likes to fit the noise with a single cluster with very large covariance
# Need to work on this, or figure out what is noise ahead of time and only pass datapoints to EM_GMM that we think are real cluster data

noise_points = 40

range_min = min(min(clean_obs[:,0]),min(clean_obs[:,1]))
range_max = max(max(clean_obs[:,0]),max(clean_obs[:,1]))
data_range = 2.0*max(abs(range_min),range_max)
noise = (random.rand(noise_points,2)-0.5)*data_range


# Choose to use either the clean data, or the data + noise
#obs = vstack((clean_obs,noise))
obs = clean_obs

# Plot the initial data

fignum=0

fignum=fignum+1
pylab.figure(fignum)

pylab.plot(obs[:,0],obs[:,1],'k.',markersize=2.0)
pylab.plot(x,y,'bo')

pylab.savefig('initial_data.png')

# initialize centroids for EM

#k -= 1   # test overfitting
#k += 1   # test underfitting


# For ten trials, use k-means to generate centroids, and let EM iterate to the best fit parameters.
# The function 'EM_GMM' is what does all the work.

# Repeating the process for a number of independent trials gives us a sense of the likelihood space.  Are there many local maximia?
# Is it smooth?

# For each step update the maximum likelihood if we find a new max, save this for later.
# (To do: save the parameters from each trial so we don't have to repeat things later.)
# Note that the likelihoods returned by EM_GMM are large negative numbers.

maxLike = -inf
for i in range(10):

    centroids, labels = kmeans2(obs,k,minit='points')
    Mu = centroids.T

    mu, Sigma, LL = EM_GMM(obs,k,Mu,max_iter=10)

    print 'Likelihood is: ', LL[-1]

    maxLike = max(LL[-1],maxLike)


print 'Maximum Likelihood is: ', maxLike
print

# Now that we have a sense of the global maximum likelihood, repeat the process until we match or beat that score.

trial_likelihood = -inf
while trial_likelihood < 1.01*maxLike:

    centroids, labels = kmeans2(obs,k,minit='points')
    Mu = centroids.T

    # Add some jitter to the centroids before passing them to E_M; this helps us break out of local maxima that trap k-means.

    centroid_jitter = random.standard_normal(shape(Mu))

    #mu, Sigma, LL = EM_GMM(obs,k,Mu,max_iter=20)
    mu, Sigma, LL = EM_GMM(obs,k,Mu+centroid_jitter,max_iter=20)

    trial_likelihood = LL[-1]

    print 'Trial Likelihood: ', trial_likelihood
    

print 'Likelihood has passed!'
print
print 'Iterations: ', len(LL)
print
print 'Trial Likelihoods: ', LL
print

#print Mu
#print
#print Mu+centroid_jitter
#print
#print mu
#print
#print x
#print y


# Now update the plot of the data with contours of the found cluster parameters.  This takes a bit of time.

delta = 0.2
a = arange(min(obs[:,0]), max(obs[:,0]), delta)
b = arange(min(obs[:,1]), max(obs[:,1]), delta)
X, Y = meshgrid(a, b)

# standard deviations for a gaussian, 1-3 sigma
sigma_spacing = array([1., 2., 3.])
sigma_levels = 1/sqrt(2*pi) * exp(-sigma_spacing**2/2.0)

#L1 = zeros(shape(X))
#L2 = zeros(shape(X))
#L3 = zeros(shape(X))

LL = zeros(shape(X))

matplotlib.rcParams['contour.negative_linestyle'] = 'solid'

rows, columns = shape(X)

for kk in range(k):
    for i in range(rows):
        for j in range(columns):
            pos = array([X[i,j], Y[i,j]])

            LL[i,j] = MVNLikelihood(pos,mu[:,kk],Sigma[kk,:,:])

    CS1 = pylab.contour(X, Y, LL, sigma_levels, linewidths=2.0)

# Mark the found centroids with red stars
pylab.plot(mu[0,:],mu[1,:],'r*',markersize=10.0)

pylab.grid(True)
#pylab.xlim(1.0,6.0)
#pylab.ylim(0.5,5.5)

pylab.savefig('initial_data.png')


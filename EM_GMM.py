#! /usr/bin/env python
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This function implements the expectation-maximization algorithm for two 2D MVN distributions
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from numpy import *
import time

# EM algorithm for multivariate normal distributions
# The number of clusters k is an input
# The initial cluster centroids are an input

# The observation vector must have the shape [N, dim]
# The input centroids mu must have the shape [dim, k], where mu[:,0] are the coordinates for the first centroid, etc

# Function returns means, covariances of final MVN distributions, and vector of likelihood calculated after each iteration

def EM_GMM(obs,k,mu,max_iter=20):

    # Helper function for MVN PDF, returns either scalar likelihood if input is 
    # a single point, or a vector of likelihoods if the input is an array of points
    def MVNLikelihood(x,mu,cov,N,dim):
        if N==1:
            L = power(2*pi,-dim/2.0) * 1.0/sqrt(linalg.det(cov)) * exp(-0.5*dot(x-mu,dot(linalg.inv(cov),x-mu)))
        else:
            L = empty(N)

            # A few declarations to speed up computation
            determinant = 1.0/sqrt(linalg.det(cov))
            prefactor = power(2*pi,-dim/2.0) * determinant
            inverse = linalg.inv(cov)
            subMeans = x - tile(mu,(N,1))
            for ii in range(N):
                L[ii] = prefactor * exp(-0.5*dot(subMeans[ii,:],dot(inverse,subMeans[ii,:])))
        return L

    # Helper function for weighted MVN likelihood
    # basically just unpacks arrays for weights, means and covariances
    # returns a vector, with sum of likelihoods for each point
    def PDFnormalize(x,tau,mu,Sigma,k,N,dim):
        normalization_factor = zeros(N)
        for jj in range(N):
            for kk in range(k):
                normalization_factor[jj] += tau[kk]*MVNLikelihood(obs[jj,:], mu[:,kk], Sigma[kk,:,:],1,dim)
        return normalization_factor

    # Initialize a list to collect likelihood after each interation
    LL = []

    # get the dimensions of the dataset
    N, dim = shape(obs)

    # Factor to prevent division by zero
    TINY = 1e-6

    # Generate an array of identity matrices for the initial covariance matrices
    Sigma = 3.0*ones((k, dim, dim))*identity(dim)

    # initialize array of mixture parameters, with flat distribution to start
    tau = ones(k)/k

    # Initialize the responsibility function (or membership probabilities; called gamma in Bishop Sec. 9.2)...
    # should rename this, don't confuse it with '.T' which is numpy transpose
    T = zeros((k,N))

    # ...and a counter for the (weighted) number of members of each cluster:
    N_cluster = zeros(k)

    # Initialize likelihoods
    for i in range(k):
        
        # Calculate likelihoods for every point with every cluster
        T[i,:] = MVNLikelihood(obs, mu[:,i], Sigma[i,:,:],N,dim)

    # Generate matrics that store normalization factors for each cluster
    Tau = tile(tau,(N,1)).T
    normalization_factor = tile(sum(Tau*T,0),(k,1))

    LL.append(sum(log(normalization_factor)))

    # Start the EM loop
    counter = 0
    EStep_time = 0
    MStep_time = 0
    LStep_time = 0
    converged = False

    while not converged:

        ### E-STEP
        ### in which we calculate the expectation for each data point to belong to each cluster

        ### We use the likelihoods calculated for the convergence step

        clock1 = time.time()

        T = Tau * T / normalization_factor

        clock2 = time.time()
        EStep_time += clock2-clock1

        clock1 = time.time()

        ### M-STEP
        # in which we use the maximum likelihood formalism to update the cluster means, 
        # covariances, and mixing coefficients, based on the weights calculated in the E-step
        for i in range(k):

            # Update the number of points in each cluster, this is used as a normalization to the updated mean
            N_cluster[i] = T[i,:].sum()
            
            # update the overall probability for this cluster
            tau[i] = N_cluster[i]/N

            # Make a copy of the weights across each dimension
            W = tile(T[i,:],(dim,1)).T

            # update the mean for this cluster
            mu[:,i] = sum(W*obs,0)/N_cluster[i]

            # update the covariance matrix for this cluster
            Sigma[i,:,:] = (dot((obs-mu[:,i]).T, W*(obs-mu[:,i]).conj())) / N_cluster[i]


        clock2 = time.time()
        MStep_time += clock2-clock1


        ### TEST FOR CONVERGENCE
        # in which we sum the log-likelihood of each point over each cluster, using the weights.
        # This should be a monotonically increasing value as the algorithm proceeds.

        clock1 = time.time()

        for i in range(k):

            # Calculate likelihoods for every point with every cluster
            T[i,:] = MVNLikelihood(obs, mu[:,i], Sigma[i,:,:],N,dim)

        Tau = tile(tau,(N,1)).T
        normalization_factor = tile(sum(Tau*T,0),(k,1))

        LL.append(sum(log(normalization_factor)))

        clock2 = time.time()
        LStep_time += clock2-clock1

        #print
        #print 'Iteration: ', counter
        #print 'Marginalized log-likelihood:', LL[-1]

        if counter > 1 and LL[-1] - LL[counter-2] < 0.0:
            print 'Log-likelihood has decreased!'
            print 'Previous step: ', LL[-2]
            print 'Current step: ', LL[-1]

        # Convergence check
        # If the likelihood is unchanged at the 0.01 level then stop the iteration
        # This should be fine-tuned for number of points, clusters, dimensionality, etc
        converged = (LL[-1] - LL[-2]) <= 1e-2 or counter >= max_iter
        counter += 1

    #print
    #print 'Algorithm has converged!'

    #print 'Time in E-step: ', EStep_time
    #print 'Time in M-step: ', MStep_time
    #print 'Time in L-step: ', LStep_time

    return mu, Sigma, LL

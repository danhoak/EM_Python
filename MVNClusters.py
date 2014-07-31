#! /usr/bin/env python

from numpy import *
#import pylab

# This function returns a set of synthetic observations in a 2-dimensional feature space
# The observations belong to 2-dimensional MVN distributions

# Inputs:
# 'number' is the number of clusters, N
# 'size1' is the lower limit on the number of members in a cluster
# 'size2' is the upper limit on the number of members in a cluster
# 'background' is a string ("off" or "on") that enables/disables background noise

# Outputs:
# 'obs' is an Nx2 array of observations
# 'x' is a 1xN array of x-coordinates of centroids
# 'y' is a 1xN array of y-coordinates of centroids
# 'labels' is a 1xN array of cluster labels in the range [0,N-1]

# The span of the observation space is -15 to 15.

def MVNClusters(number=5,size1=100,size2=100,background='off'):

    obs = zeros((number*size1,2))
    labels = zeros((number*size1))


    # Generate a 1xN array of random cluster orientations

    theta = random.rand(number)*2*pi

    # Generate 1xN arrays of x-coords and y-coords
    x = 20.0*(random.rand(number)-0.5)
    y = 20.0*(random.rand(number)-0.5)

    covariances = random.rand(number)

    for i in range(number):

        rotation = array([[cos(theta[i]),-sin(theta[i])],[sin(theta[i]),cos(theta[i])]])
        covariance = array([[1.0,covariances[i]],[covariances[i],1.0]])

        centroid = array([x[i],y[i]])

        r = random.multivariate_normal(zeros((2)),covariance,size1)

        for j in range(size1):

            obs[size1*i+j] = dot(r[j,:],rotation) + centroid

            labels[size1*i+j] = i

    return obs, x, y, labels

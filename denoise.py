"""
ECE 4110 Project 1 
Name: Yiqi Jiang, Huihua Yu
NetID: yj89, hy437
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
X=np.load('noisy.npy')

# Calculate expectation of X by using Law of Large Number
n = 5421
expX = np.average(X, axis=0)

# Calculate expectation of Z by using the expectation of X
temp1 = np.array([expX[0], expX[1], expX[2], expX[3], expX[25], expX[26], expX[27]])
temp2 = np.array([expX[:,0][4:25], expX[:,1][4:25], expX[:,2][4:25], expX[:,3][4:25], expX[:,-4][4:25], expX[:,-3][4:25], expX[:,-2][4:25], expX[:,-1][4:25]])

numrow = np.sum(np.sum(temp1, axis=0))
numcol = np.sum(np.sum(temp2, axis=0))

e_z = (numrow+numcol)/(7*28+8*21)

expZ = np.repeat(e_z, 784).reshape(28,28)

# Calculate the covariance matrix of noisy image X
sigmaX = np.zeros((784,784))
for i in range(0,5421):
    delta = (X[i]-expX).reshape(1,-1)
    sigmaX += 1/5421*np.matmul(delta.T, delta)
sigmaX.shape

# Calculate cov-distance function
diffZ = X - e_z
diffZ_vector = np.reshape(diffZ, (len(diffZ),-1))
mean_diffZ_cross_product = np.zeros((diffZ_vector.shape[1], diffZ_vector.shape[1]))
for i in range(len(X)):
    mean_diffZ_cross_product += np.outer(diffZ_vector[i], diffZ_vector[i])/len(diffZ)

numdis = np.zeros(55)
discov0 = np.zeros(55)
avoid = [4, 24]
for i in range(28):
    for j in range(28):
        for k in range(28):
            for l in range(28):
                if not (avoid[0]<=i<=avoid[1] and avoid[0]<=j<=avoid[1]) \
                    and not (avoid[0]<=k<=avoid[1] and avoid[0]<=l<=avoid[1]):
                    d = np.abs(i-k)+np.abs(j-l)
                    discov0[d] += mean_diffZ_cross_product[28*i+j, 28*k+l]
                    numdis[d] += 1

discov = discov0/numdis

# Calculate the covariance matrix of noise Z
sigmaZ = np.zeros((784,784))

for i in range(0,28):
    for j in range(0,28):
        for k in range(0,28):
            for l in range(0,28):
                d = np.abs(i-k)+np.abs(j-l)
                sigmaZ[i*28+j][k*28+l] = discov[d]

# Denoise each image with linear estimation
expX_f = expX.flatten()
X_f = X.reshape((len(X), -1))
deltaZ = np.matmul(np.matmul(sigmaZ, np.linalg.inv(sigmaX)), (X_f-expX_f).T)
deltaZ = deltaZ.T.reshape(-1,28,28)
denoised_X = X - expZ - deltaZ
denoised_X = np.clip(denoised_X, 0, 255)

np.save('denoised.npy', denoised_X)

"""
fig = plt.figure()
plt.gray()  # show the filtered result in grayscale
ax1 = fig.add_subplot(131)  # left side
ax2 = fig.add_subplot(132)  # middle
ax3 = fig.add_subplot(133)  # right side
ax1.imshow(X[10], cmap='gray', vmin=0, vmax=255)
ax2.imshow(denoised_X[10], cmap='gray', vmin=0, vmax=255)
ax3.imshow(denoised_X[5], cmap='gray', vmin=0, vmax=255)
plt.show()
"""
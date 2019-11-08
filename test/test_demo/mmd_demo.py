import torch
import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
from mpl_toolkits.mplot3d import Axes3D
from test import mmd

SAMPLE_SIZE = 500
buckets = 50

mu = -0.6
sigma = 0.15
res1 = []
mean1 = [0, 0]
cov1 = [[1, 0], [0, 1]]
mean2 = [-5, -5]
cov2 = [[1, 0], [0, 1]]

# Draw the distribution
'''
x = np.linspace(-5,5,100)
y = np.linspace(-5,5,100)
X, Y = np.meshgrid(x,y)
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X; pos[:, :, 1] = Y
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, rX.pdf(pos),cmap='viridis',linewidth=0)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.show()
'''

same_1 = []
for i in range(10):
    same_1.append([np.random.multivariate_normal(mean1, cov1).tolist() for _ in range(1, SAMPLE_SIZE)])

same_2 = []
for i in range(10):
    same_2.append([np.random.multivariate_normal(mean1, cov1).tolist() for _ in range(1, SAMPLE_SIZE)])
X = torch.Tensor(same_1)
Y = torch.Tensor(same_2)
X, Y = Variable(X), Variable(Y)
print("Same Distribution: " + str(mmd.mmd_rbf(X, Y)))

diff_1 = []
for i in range(10):
    diff_1.append([np.random.multivariate_normal(mean1, cov1).tolist() for _ in range(1, SAMPLE_SIZE)])

diff_2 = []
for i in range(10):
    diff_2.append([np.random.multivariate_normal(mean2, cov2).tolist() for _ in range(1, SAMPLE_SIZE)])
X = torch.Tensor(diff_1)
Y = torch.Tensor(diff_2)
X, Y = Variable(X), Variable(Y)
print("Different Distribution: " + str(mmd.mmd_rbf(X, Y)))

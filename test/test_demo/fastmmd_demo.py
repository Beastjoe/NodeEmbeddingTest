import matlab.engine
import matlab
import numpy as np
import config
import matplotlib.pyplot as plt

eng = matlab.engine.start_matlab()
eng.addpath(config.FASTMMD_MATLAB_PATH, nargout=0)

sigma = []
for i in np.linspace(-2, 2, 21):
    sigma.append(10 ** float(i))
SAMPLE_SIZE = 500
mu = -0.6
res1 = []
mean1 = [0, 0, 0]
cov1 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
mean2 = [-5, -5, -5]
cov2 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

same_1 = [np.random.multivariate_normal(mean1, cov1).tolist() for _ in range(1, SAMPLE_SIZE)]
same_2 = [np.random.multivariate_normal(mean1, cov1).tolist() for _ in range(1, SAMPLE_SIZE)]
sigma_matlab = matlab.double(sigma)
same_1_matlab = matlab.double(same_1)
same_2_matlab = matlab.double(same_2)
d1, f1 = eng.MMD3(same_1_matlab, same_2_matlab, sigma_matlab, nargout=2)
plt.subplot(1,2,1)
plt.plot(sigma, d1, label='MMD-biased')
plt.plot(sigma, f1, label='MMD-unbiased')
plt.xlabel("Same Distribution")
plt.legend()
plt.xscale("log")

diff_1 = [np.random.multivariate_normal(mean1, cov1).tolist() for _ in range(1, SAMPLE_SIZE)]
diff_2 = [np.random.multivariate_normal(mean2, cov2).tolist() for _ in range(1, SAMPLE_SIZE)]
sigma_matlab = matlab.double(sigma)
diff_1_matlab = matlab.double(diff_1)
diff_2_matlab = matlab.double(diff_2)
d2, f2 = eng.MMD3(diff_1_matlab, diff_2_matlab, sigma_matlab, nargout=2)
plt.subplot(1,2,2)
plt.plot(sigma, d2, label='MMD-biased')
plt.plot(sigma, f2, label='MMD-unbiased')
plt.xlabel("Different Distribution")
plt.legend()
plt.xscale("log")
plt.show()

print(d1)

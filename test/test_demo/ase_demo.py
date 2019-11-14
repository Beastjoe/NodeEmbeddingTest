import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from graspy.inference import LatentDistributionTest
from graspy.embed import AdjacencySpectralEmbed
from graspy.simulations import sbm, rdpg
from graspy.utils import symmetrize
from graspy.plot import heatmap, pairplot

n_components = 4 # the number of embedding dimensions for ASE
P = np.array([[0.8, 0.2, 0.6, 0.5],
              [0, 0.9, 0.3, 0.2],
              [0, 0, 0.5, 0.2],
              [0, 0, 0, 0.5]])

P = symmetrize(P)
csize = [50] * 4
A = sbm(csize, P)
X = AdjacencySpectralEmbed(n_components=n_components).fit_transform(A)


A1 = sbm(csize, P)
#heatmap(A1, title='4-block SBM adjacency matrix A1')
X1 = AdjacencySpectralEmbed(n_components=n_components).fit_transform(A1)

P2 = np.array([[0.8, 0.2, 0.2, 0.5],
              [0, 0.9, 0.3, 0.2],
              [0, 0, 0.5, 0.2],
              [0, 0, 0, 0.5]])

P2 = symmetrize(P2)
A2 = sbm(csize, P2)

ldt = LatentDistributionTest()
p = ldt.fit(A2, A1)

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(ldt.null_distribution_, 50)
ax.axvline(ldt.sample_T_statistic_, color='r')
ax.set_title("P-value = {}".format(p), fontsize=20)
plt.savefig('demo.png', bbox_inches='tight')
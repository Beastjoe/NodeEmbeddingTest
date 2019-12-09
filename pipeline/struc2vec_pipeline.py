import os
import subprocess
# import matlab.engine
# import matlab
import config as config
import numpy as np
from dataset import synthetic_generator
# import test.mmd as mmd
# import test.fastmmd as fastmmd
# import test.ase as ase
from graspy.inference import LatentDistributionTest
from graspy.embed import AdjacencySpectralEmbed
from graspy.simulations import sbm, rdpg
from graspy.utils import symmetrize
from graspy.plot import heatmap, pairplot
import matplotlib.pyplot as plt
import networkx as nx


def prepare_data(filename, sizes, probs, dataset='er'):
    """
    :param filename: filename of prepared dataset
    :param sizes: size param for dataset generator
    :param probs: probability param for dataset generator
    :param dataset: testing dataset. Default is erdos renyi graph.
    :return:
    """
    if dataset == 'er':
        graph = synthetic_generator.generate_erdos_renyi_graph(sizes, probs)
    elif dataset == 'block':
        graph = synthetic_generator.generate_stochastic_block_model(sizes, probs)
    elif dataset == 'kronecker':
        graph = synthetic_generator.generate_stochastic_kronecker_graph(sizes, probs)

    input_file_path = os.path.join(config.INPUT_PATH, filename + '.edgelist')
    with open(input_file_path, 'w') as f:
        for e in graph.edges():
            f.write(str(e[0]) + ' ' + str(e[1]) + '\n')


def run_embedding(input_file_path, output_path=config.OUTPUT_PATH, num_walks=20, walk_length=80, window_size=5, dim=2,
                  opt1=True,
                  opt2=True, opt3=True, until_layer=6):
    output_filename = os.path.split(input_file_path)[-1].split('.')[0] + '_128d_struct2vec' + '.emb'

    commands = ['python', config.STRUC2VEC_MAIN_PATH, '--input', input_file_path, '--output',
                os.path.join(output_path, output_filename),
                '--num-walks', str(num_walks), '--walk-length', str(walk_length), '--window-size', str(window_size),
                '--dimensions', str(dim), '--OPT1', str(opt1), '--OPT2', str(opt2), '--OPT3', str(opt3),
                '--until-layer',
                str(until_layer)]
    to_print = "Executing command:\n"
    for item in commands:
        to_print += item + ' '
    to_print += '\n'
    print(to_print)
    subprocess.call(commands)


def run_test(input_source_file_path, intput_target_file_path, sample_size, batch_size, method='mmd'):
    source_list = []
    target_list = []
    with open(input_source_file_path, 'r') as f:
        line = f.readline()
        dim = int(line.split(' ')[1])
        line = f.readline()
        while line:
            node = []
            represents = line.split(' ')
            for i in range(1, dim + 1):
                node.append(float(represents[i]))
            source_list.append(node)
            line = f.readline()

    with open(intput_target_file_path, 'r') as f:
        f.readline()
        line = f.readline()
        while line:
            node = []
            represents = line.split(' ')
            for i in range(1, dim + 1):
                node.append(float(represents[i]))
            target_list.append(node)
            line = f.readline()

    if method == 'mmd':
        sigma = []
        for i in np.linspace(-2, 2, 21):
            sigma.append(10 ** float(i))
        fastmmd.mmd_test(source_list, target_list, sigma, 1024, 1000)
        # mmd.mmd_test(source_list, target_list, batch_size, sample_size)
    elif method == 'fastmmd':
        sigma = []
        for i in np.linspace(-2, 2, 21):
            sigma.append(10 ** float(i))
        fastmmd.mmd_test(source_list, target_list, sigma, 1024, 500)
    elif method == 'ase':
        ase.ase_test(source_list, target_list, sample_size, 200)



for i in range(10, 20):
    # prepare_data('block-3_' + str(i), [100, 100, 300], [[0.75, 0.05, 0.05], [0.05, 0.75, 0.05], [0.05, 0.05, 0.75]],
    #              dataset='block')
    # prepare_data('block-1_' + str(i), [100, 100, 300], [[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]],
    #              dataset='block')
    prepare_data('er-05_' + str(i), 100, 0.5,
                 dataset='er')
    prepare_data('er-005_' + str(i), 100, 0.35,
                 dataset='er')
    # prepare_data('kronecker-1_' + str(i), 9, np.array([[0.98, 0.58], [0.58, 0.6]]), dataset='kronecker')
    # prepare_data('kronecker-2_' + str(i), 9, np.array([[0.5, 0.8], [0.8, 0.9]]), dataset='kronecker')

# for i in range(10):
#     run_embedding(os.path.join(config.INPUT_PATH, 'block-3_' + str(i) + '.edgelist'))
#     run_embedding(os.path.join(config.INPUT_PATH, 'block-1_' + str(i) + '.edgelist'))
#     run_embedding(os.path.join(config.INPUT_PATH, 'er-05_' + str(i) + '.edgelist'))
#     run_embedding(os.path.join(config.INPUT_PATH, 'er-005_' + str(i) + '.edgelist'))
#     run_embedding(os.path.join(config.INPUT_PATH, 'kronecker-1_' + str(i) + '.edgelist'))
#     run_embedding(os.path.join(config.INPUT_PATH, 'kronecker-2_' + str(i) + '.edgelist'))

# for i in range(10):
#    for j in range(10):
#        run_test(os.path.join(config.OUTPUT_PATH, 'block-1_'+str(i)+'.emb'), os.path.join(config.OUTPUT_PATH, 'block-3_'+str(j)+'.emb'), 2000, 1)

for i in range(10, 20):
    for j in range(10, 20):
        G_er = nx.read_edgelist("../input/er-05_{}.edgelist".format(i))
        G_k = nx.read_edgelist("../input/er-005_{}.edgelist".format(j))

        ldt = LatentDistributionTest()
        p = ldt.fit(G_er, G_k)
        print(p)

# fig, ax = plt.subplots(figsize=(10, 6))
# ax.hist(ldt.null_distribution_, 50)
# ax.axvline(ldt.sample_T_statistic_, color='r')
# ax.set_title("P-value = {}".format(p), fontsize=20)
# plt.show();


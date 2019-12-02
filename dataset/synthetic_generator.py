import networkx as nx
import random
from dataset import KroneckerGenerator
from dataset.KroneckerInitMatrix import InitMatrix


def generate_stochastic_block_model(sizes, probs):
    """

    :param sizes: sizes of each blocks
    :param probs: list of list of floats. Element(i, j) gives the density of edges
                going from nodes of group i to nodes of group j.
    :return: Stochastic block model NetworkX Graph
    """
    return nx.stochastic_block_model(sizes, probs, seed=0)


def generate_erdos_renyi_graph(size, prob):
    """

    :param size: size of nodes
    :param prob: probability of a edge between node pairs
    :return: Erdos-Renyi graph NetworkX Graph
    """
    return nx.gnp_random_graph(size, prob)


def generate_stochastic_kronecker_graph(k, prob):
    nodes = 2
    init = InitMatrix(nodes)
    init.make()
    init.makeStochasticCustom(prob)
    return KroneckerGenerator.generateStochasticKron(init, k, True)


def _preprocess_ca_dataset():
    with open('CA-AstroPh.txt') as fin:
        node_table = dict()
        line = fin.readline()
        fout = open('CA-AstroPh_real.edgelist', 'w')
        while line:
            if int(line.split()[0]) not in node_table:
                node_table[int(line.split()[0])] = len(node_table)
            if int(line.split()[1]) not in node_table:
                node_table[int(line.split()[1])] = len(node_table)
            fout.write(str(node_table[int(line.split()[0])]) + ' ' + str(node_table[int(line.split()[1])]) + '\n')
            line = fin.readline()
        fout.close()

    with open('CA-GrQc.txt') as fin:
        node_table = dict()
        line = fin.readline()
        fout = open('CA-GrQc_real.edgelist', 'w')
        while line:
            if int(line.split()[0]) not in node_table:
                node_table[int(line.split()[0])] = len(node_table)
            if int(line.split()[1]) not in node_table:
                node_table[int(line.split()[1])] = len(node_table)
            fout.write(str(node_table[int(line.split()[0])]) + ' ' + str(node_table[int(line.split()[1])]) + '\n')
            line = fin.readline()
        fout.close()


def simple_flip(fp_prob=0.00001, fn_prob=0.01):
    with open('CA-AstroPh.txt') as fin:
        node_table = dict()
        edge_table = dict()
        line = fin.readline()
        fout = open('CA-AstroPh_flip_' + str(fp_prob) + '_' + str(fn_prob) + '.edgelist', 'w')
        while line:
            if int(line.split()[0]) not in node_table:
                node_table[int(line.split()[0])] = len(node_table)
                edge_table[int(line.split()[0])] = set()
            if int(line.split()[1]) not in node_table:
                node_table[int(line.split()[1])] = len(node_table)
                edge_table[int(line.split()[1])] = set()
            edge_table[int(line.split()[0])].add(int(line.split()[1]))
            edge_table[int(line.split()[1])].add(int(line.split()[0]))

            line = fin.readline()
        new_edge_table = dict()
        for node1 in node_table.items():
            new_edge_table[node1[0]] = set()
            print(node1[1] / len(node_table))
            for node2 in node_table.items():
                if node2[0] in new_edge_table:
                    if node1[0] in new_edge_table[node2[0]]:
                        new_edge_table[node1[0]].add(node2[0])
                    continue

                if node1 != node2:
                    if node2[0] in edge_table[node1[0]]:
                        if random.random() < fn_prob:
                            continue
                        else:
                            new_edge_table[node1[0]].add(node2[0])
                    else:
                        if random.random() < fp_prob:
                            new_edge_table[node1[0]].add(node2[0])

        for edges in new_edge_table.items():
            if len(edges[1]) == 0:
                continue
            else:
                for node in edges[1]:
                    fout.write(
                        str(node_table[edges[0]]) + ' ' + str(node_table[node]) + '\n')
        fout.close()

    with open('CA-GrQc.txt') as fin:
        node_table = dict()
        line = fin.readline()
        fout = open('CA-GrQc_flip_' + str(fp_prob) + '_' + str(fn_prob) + '.edgelist', 'w')
        while line:
            if int(line.split()[0]) not in node_table:
                node_table[int(line.split()[0])] = len(node_table)
                edge_table[int(line.split()[0])] = set()
            if int(line.split()[1]) not in node_table:
                node_table[int(line.split()[1])] = len(node_table)
                edge_table[int(line.split()[1])] = set()
            edge_table[int(line.split()[0])].add(int(line.split()[1]))
            edge_table[int(line.split()[1])].add(int(line.split()[0]))

            line = fin.readline()
        new_edge_table = dict()
        for node1 in node_table.items():
            new_edge_table[node1[0]] = set()
            print(node1[1]/len(node_table))
            for node2 in node_table.items():
                if node2[0] in new_edge_table:
                    if node1[0] in new_edge_table[node2[0]]:
                        new_edge_table[node1[0]].add(node2[0])
                    continue

                if node1 != node2:
                    if node2[0] in edge_table[node1[0]]:
                        if random.random() < fn_prob:
                            continue
                        else:
                            new_edge_table[node1[0]].add(node2[0])
                    else:
                        if random.random() < fp_prob:
                            new_edge_table[node1[0]].add(node2[0])

        for edges in new_edge_table.items():
            if len(edges[1]) == 0:
                continue
            else:
                for node in edges[1]:
                    fout.write(
                        str(node_table[edges[0]]) + ' ' + str(node_table[node]) + '\n')
        fout.close()


#simple_flip(0, 0.01)
#simple_flip(0, 0.2)
#simple_flip(0, 0.3)
#simple_flip(1e-5, 0.05)
#simple_flip(1e-5, 0.1)
#simple_flip(1e-5, 0.2)
#simple_flip(1e-5, 0.3)
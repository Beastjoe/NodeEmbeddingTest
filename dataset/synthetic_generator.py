import networkx as nx


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

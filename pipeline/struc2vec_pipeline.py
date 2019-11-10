import os
import subprocess
import config as config
from dataset import synthetic_generator
import test.mmd as mmd


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

    input_file_path = os.path.join(config.INPUT_PATH, filename + '.edgelist')
    with open(input_file_path, 'w') as f:
        for e in graph.edges():
            f.write(str(e[0]) + ' ' + str(e[1]) + '\n')


def run_embedding(input_file_path, output_path=config.OUTPUT_PATH, num_walks=20, walk_length=80, window_size=5, dim=2,
                  opt1=True,
                  opt2=True, opt3=True, until_layer=6):
    output_filename = os.path.split(input_file_path)[-1].split('.')[0] + '.emb'

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
        mmd.mmd_test(source_list, target_list, batch_size, sample_size)


# prepare_data('block-3', [250, 250, 1500], [[0.75, 0.05, 0.05], [0.05, 0.75, 0.05], [0.05, 0.05, 0.75]], dataset='block')
# prepare_data('block-1', [250, 250, 1500], [[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]], dataset='block')

# run_embedding(os.path.join(config.INPUT_PATH, 'block-3.edgelist'))
# run_embedding(os.path.join(config.INPUT_PATH, 'block-1.edgelist'))

run_test(os.path.join(config.OUTPUT_PATH, 'block-1.emb'), os.path.join(config.OUTPUT_PATH, 'block-1.emb'), 500, 10)

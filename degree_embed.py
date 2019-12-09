import torch
from os import listdir
from os.path import isfile, join
import numpy as np

onlyfiles = [f for f in listdir('input') if isfile(join('input', f))]
for name in onlyfiles:
	f = open('input/' + name, 'r', encoding = 'unicode_escape')
	print(name)
	num_node = 0
	if name[:3] == 'blo' or name[:3] == 'er-':
		num_node = 500
	elif name[:3] == 'kro':
		num_node = 512
	elif name[:4] == 'CA-A':
		num_node = 19000
	else:
		num_node = 5600
	edge_deg = torch.zeros(num_node)
	contents = f.read()
	edges = contents.split('\n')
	edges.pop()

	for edge in edges:
		nodes = edge.split()
		edge_deg[int(nodes[0])] += 1.0
		edge_deg[int(nodes[1])] += 1.0

	f.close()
	f = open('outputdeg/' + name[:-9] + '.emb', 'a')
	idx_output = np.random.permutation(num_node)
	for i in range(num_node):
		curr_idx = idx_output[i]
		f.write(str(curr_idx) + ' ')
		f.write(str(edge_deg[curr_idx].item()))
		f.write('\n')
	f.close()



	#node_deg = edges[]
	# for edge in edges:
	# 	pass
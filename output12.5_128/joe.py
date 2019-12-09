import os
import subprocess
import matlab
import config as config
import numpy as np
import test.mmd as mmd
import test.fastmmd as fastmmd

def run_test(input_source_file_path, intput_target_file_path, sample_size, batch_size, method='mmd', normalized=False):
    source_list = []
    target_list = []
    smax_list = np.array([-1e9, -1e9])
    smin_list = np.array([1e9, 1e9])
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

    tmax_list = np.array([-1e9, -1e9])
    tmin_list = np.array([1e9, 1e9])
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

    if normalized:
        for s in source_list:
            for i in range(dim):
                if float(s[i]) > smax_list[i]:
                    smax_list[i] = float(s[i])
                if float(s[i]) < smin_list[i]:
                    smin_list[i] = float(s[i])
        for s in source_list:
            for i in range(dim):
                s[i] = (s[i] - smin_list[i]) / (smax_list[i] - smin_list[i])
    if normalized:
        for s in target_list:
            for i in range(dim):
                if float(s[i]) > tmax_list[i]:
                    tmax_list[i] = float(s[i])
                if float(s[i]) < tmin_list[i]:
                    tmin_list[i] = float(s[i])
        for s in target_list:
            for i in range(dim):
                s[i] = (s[i] - tmin_list[i]) / (tmax_list[i] - tmin_list[i])
        source_list, target_list = ase.median_heuristic(np.array(source_list), np.array(target_list))
        source_list = np.ndarray.tolist(source_list)
        target_list = np.ndarray.tolist(target_list)
    '''
    xs_value = []
    xt_value = []
    ys_value = []
    yt_value = []
    for s in source_list:
        xs_value.append(s[0])
        ys_value.append(s[1])
    for t in target_list:
        xt_value.append(t[0])
        yt_value.append(t[1])
    plt.subplot(1,2,1)
    plt.scatter(xs_value, ys_value)
    plt.subplot(1,2,2)
    plt.scatter(xt_value, yt_value)
    plt.show()
    '''
    if method == 'mmd':
        sigma = []
        for i in np.linspace(-2, 2, 21):
            sigma.append(10 ** float(i))
        return fastmmd.mmd_test(source_list, target_list, sigma, 1024, sample_size, method="FastMMD-Fastfood")
        # mmd.mmd_test(source_list, target_list, batch_size, sample_size)
    elif method == 'fastmmd':
        sigma = []
        for i in np.linspace(-2, 2, 21):
            sigma.append(10 ** float(i))
        fastmmd.mmd_test(source_list, target_list, sigma, 1024, sample_size, method="MMD-linear")
    elif method == 'ase':
        ase.ase_test(source_list, target_list, sample_size, 200)




fpr_tpr = np.zeros((2, 10000))
i = 0
same_distribution = []
diff_distribution = []
sz = 10
for i in range(sz):
    for j in range(sz):
        val = np.max(run_test(os.path.join(config.OUTPUT_PATH,'kronecker-1_' + str(i) + '.emb'),
                       os.path.join(config.OUTPUT_PATH,'block-3_' + str(j) + '.emb'), 500, 1,
                       method='mmd'))
        diff_distribution.append(val)
for i in range(sz):
    for j in range(sz):
        res = run_test(os.path.join(config.OUTPUT_PATH, 'block-3_' + str(i) + '.emb'),
                       os.path.join(config.OUTPUT_PATH, 'block-3_' + str(j) + '.emb'), 500, 1,
                       method='mmd')
        val = np.partition(res.flatten(), -2)[-3]
        #val = np.max(run_test(os.path.join(config.OUTPUT_PATH,'block-3_' + str(i) + '.emb'),
        #               os.path.join(config.OUTPUT_PATH,'block-3_' + str(j) + '.emb'), 500, 1,
        #               method='mmd', normalized=False))
        same_distribution.append(val)
cnt = 0
score = 0.0
for threshold in np.linspace(0.0, 1.2, 10000):
    tp = 0.0
    fp = 0.0
    for i in range(sz):
        for j in range(sz):
            val = same_distribution[i * sz + j]
            if val <= threshold:
                tp += 1.0
    for i in range(sz):
        for j in range(sz):
            val = diff_distribution[i * sz + j]
            if val <= threshold:
                fp += 1.0
    fpr_tpr[0][cnt] = fp / (sz * sz)
    fpr_tpr[1][cnt] = tp / (sz * sz)
    if cnt==0:
        score += fpr_tpr[0][cnt] * fpr_tpr[1][cnt]
    else:
        score += (fpr_tpr[0][cnt] - fpr_tpr[0][cnt - 1])*fpr_tpr[1][cnt]
    cnt+=1
print(score)
#sortedArr = fpr_tpr[:, fpr_tpr[0, :].argsort()]
plt.plot(fpr_tpr[0], fpr_tpr[1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('AUCROC of struct2vec= '+str(score))
plt.savefig('aucroc_struct.png')
plt.show()
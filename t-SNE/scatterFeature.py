# -*- coding: utf-8 -*-
# python 3.5.2 with Anaconda3 4.2.0

import glob

import numpy as np
import bhtsne  # 'pip install bhtsne'

from Common import flatten, c_cycle, FeatureExtractor


def compress_to_2dim(np_mat, split_pos_list, seed=-1, perplexity=30.0):
    np_mat = np.asarray(np_mat, dtype=np.float64)
    np_mat = bhtsne.tsne(data=np_mat, dimensions=2, perplexity=perplexity, theta=0.5, rand_seed=seed)
    return np.split(np_mat, split_pos_list, axis=0)


def plot_vec(domain_vec_list, save_file='figure'):
    import matplotlib.pyplot as plt
    for i, v in enumerate(domain_vec_list):
        plt.scatter(v[1][:, 0], v[1][:, 1], label=domain_vec_list[i][0], s=40, c=c_cycle[i])
    plt.legend(loc="best")
    plt.subplots_adjust(left=0.06, bottom=0.05, right=0.985, top=0.97)
    plt.savefig(save_file)


def get_split_pos(file_len_list):
    _list = [sum(file_len_list[:n + 1]) for n, i in enumerate(file_len_list)]
    return _list[:len(file_len_list) - 1]


if __name__ == '__main__':
    class_num = 40
    model_path = "chainer_model_file.npz"
    gpu = 0
    layer = "fc8"
    size = (224, 224)
    save_name = "map.png"

    p1 = glob.glob("./Data1/*.png")
    p2 = glob.glob("./Data2/*.png")
    p3 = glob.glob("./Data3/*.png")
    lst = [["Domain1", p1], ["Domain2", p2], ["Domain3", p3]]

    # Create feature extractor
    fe = FeatureExtractor(class_num=class_num, model_path=model_path, gpu=0)

    all_path = flatten([p1, p2, p3])
    vec = list(map(lambda f: fe.get_flat_feat(f, layer), all_path))

    files_num = list(map(lambda f: len(f), [l for d, l in lst]))  # get file num each domain
    split_pos = get_split_pos(files_num)  # get split point (for np.split in compress_to_2dim())
    names = [d for d, l in lst]  # get domain name list from lst

    split_vec = compress_to_2dim(vec, split_pos_list=split_pos, seed=-1)  # Run t-SNE and split vec
    plot_vec([[d, v] for d, v in zip(names, split_vec)], save_name)

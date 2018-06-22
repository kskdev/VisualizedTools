# -*- coding: utf-8 -*-
# python 3.5.2 with Anaconda3 4.2.0

import glob
import numpy as np
import cv2
import bhtsne  # 'pip install bhtsne'

from Common.common import flatten, c_cycle


def reshape_1dim(img_paths, size=(128, 128)):
    # map ( read image → resize image → transform 1dim from images, [pathA, pathB, pathC, ...] )
    lst = list(map(lambda f: np.ravel(cv2.resize(cv2.imread(f, 1), size)), img_paths))
    return np.asarray(lst, dtype=np.float64)  # shape:(image num, ch*W*H)


def compress_to_2dim(np_mat, split_pos_list, seed=-1, perplexity=30.0):
    np_mat = np.asarray(np_mat, dtype=np.float64)
    np_mat = bhtsne.tsne(data=np_mat, dimensions=2, perplexity=perplexity, theta=0.5, rand_seed=seed)
    return np.split(np_mat, split_pos_list, axis=0)


def plot_vec(lst, save_file='figure'):
    import matplotlib.pyplot as plt
    for i, vec in enumerate(lst):
        plt.scatter(vec[1][:, 0], vec[1][:, 1], label=lst[i][0], s=40, c=c_cycle[i])
    plt.legend(loc="best")
    plt.subplots_adjust(left=0.06, bottom=0.05, right=0.985, top=0.97)
    plt.savefig(save_file)
    plt.clf()


def get_split_pos(lst):
    l = [sum(lst[:n + 1]) for n, i in enumerate(lst)]
    return l[:len(lst) - 1]


def draw_image_with_tsne(lst, save_file="figure", resize=(128, 128), seed=-1):
    files_num = list(map(lambda f: len(f), [l for d, l in lst]))
    split_pos = get_split_pos(files_num)  # get split point (for np.split)
    paths = flatten([l for d, l in lst])
    names = [d for d, l in lst]

    vec = reshape_1dim(paths, size=resize)  # Load images
    split_vec = compress_to_2dim(vec, split_pos_list=split_pos, seed=seed)  # Run t-SNE and split vec

    plot_vec([[d, v] for d, v in zip(names, split_vec)], save_file)


if __name__ == '__main__':
    p1 = glob.glob("./Data1/*.png")
    p2 = glob.glob("./Data2/*.png")
    p3 = glob.glob("./Data3/*.png")
    lst = [["Domain1", p1], ["Domain2", p2], ["Domain3", p3]]
    save_name = "map.png"

    draw_image_with_tsne(lst, save_name, resize=(128, 128))

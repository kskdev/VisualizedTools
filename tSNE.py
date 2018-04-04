# -*- coding: utf-8 -*-
# python 3.5.2 with Anaconda3 4.2.0
# coded by osmksk05

import glob
import numpy as np
import cv2
import bhtsne  # 'pip install bhtsne'
import matplotlib as mpl
import warnings

mpl.use('Agg')
warnings.filterwarnings("ignore")

c_cycle = ("#1E88E5", "#43A047", "#e53935", "#FFB300", "#616161", "#5E35B1", "#F06292", "#795548")
mpl.rc('font', size=10)
mpl.rc('lines', linewidth=2, color="#2c3e50")
mpl.rc('patch', linewidth=0, facecolor="none", edgecolor="none")
mpl.rc('text', color='#2c3e50')
mpl.rc('axes', facecolor='#ffffff', edgecolor="#111111", titlesize=15, labelsize=12, color_cycle=c_cycle, grid=False)
mpl.rc('xtick.major', size=10, width=0)
mpl.rc('ytick.major', size=10, width=0)
mpl.rc('xtick.minor', size=10, width=0)
mpl.rc('ytick.minor', size=10, width=0)
mpl.rc('ytick', direction="out")
mpl.rc('grid', color='#c0392b', alpha=0.5, linewidth=1)
mpl.rc('legend', fontsize=15, markerscale=1, labelspacing=0.2, frameon=True, fancybox=True,
       handlelength=0.1, handleheight=0.5, scatterpoints=1, facecolor="#eeeeee")
mpl.rc('figure', figsize=(10, 6), dpi=128, facecolor="none", edgecolor="none")
mpl.rc('savefig', dpi=128)


def flatten(nested_list):
    return [e for inner_list in nested_list for e in inner_list]


def reshape_1dim(img_paths, size=(128, 128)):
    # map ( read image → resize image → transform 1dim from images, [pathA, pathB, pathC, ...] )
    lst = list(map(lambda f: np.ravel(cv2.resize(cv2.imread(f, 1), size)), img_paths))
    return np.asarray(lst, dtype=np.float64)  # shape:(image num, ch*W*H)


def compress_to_2dim(vec, split_pos_list, seed=-1, perplexity=30.0):
    vec = np.asarray(vec, dtype=np.float64)
    vec = bhtsne.tsne(data=vec, dimensions=2, perplexity=perplexity, theta=0.5, rand_seed=seed)
    return np.split(vec, split_pos_list, axis=0)


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
    p1 = glob.glob("./domain_adaptation_images/amazon/images/*/*.jpg")
    p2 = glob.glob("./domain_adaptation_images/dslr/images/*/*.jpg")
    p3 = glob.glob("./domain_adaptation_images/webcam/images/*/*.jpg")

    domain_paths = [["Domain1", p1], ["Domain2", p2], ["Domain3", p3]]
    draw_image_with_tsne(domain_paths, "tSNE.png", resize=(128, 128))

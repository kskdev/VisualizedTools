# coding:utf-8

import os
import warnings

import numpy as np
import cv2
import chainer  # v2
import matplotlib as mpl

import vgg16

os.environ["PATH"] = "/usr/local/cuda-8.0/bin:/usr/bin:"
mpl.use('Agg')
warnings.filterwarnings("ignore")

# --- matplotlib setting --- #
c_cycle = ("#E51400", "#1BA1E2", "#339933", "#A200FF", "#E671B8", "#F09609", "#515151", "#A05000")
# Red, Blue, Green, Purple, Pink, Yellow, Gray, Brown
# Refer : https://matplotlib.org/users/customizing.html#matplotlib-rcparams
mpl.rc('font', size=10)
mpl.rc('lines', linewidth=2, color="#2c3e50")
mpl.rc('patch', linewidth=0, facecolor="none", edgecolor="none")
mpl.rc('text', color='#2c3e50')
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


# Convert 2dim list to 1dim list
def flatten(nested_list):
    return [e for inner_list in nested_list for e in inner_list]


def hex_to_rgb(hex_code):
    hex_code = hex_code.lstrip('#')
    lv = len(hex_code)
    return tuple(int(hex_code[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


class FeatureExtractor:
    def __init__(self, class_num, model_path, size=(224, 224), gpu=-1):
        self.class_num = class_num
        self.size = size

        self.model = vgg16.VGG16(class_num=self.class_num)
        print("--- Created network object ---")
        chainer.serializers.load_npz(model_path, self.model)
        print("--- Load network parameter ---")

        if gpu >= 0:
            chainer.cuda.get_device_from_id(gpu).use()
            self.model.to_gpu()

    # 画像の読み込み処理
    # (読み込み方法は各学習環境の入力方法に合わせた方が良い)
    def get_img(self, img_path):
        src = cv2.imread(img_path, cv2.IMREAD_COLOR)
        src = cv2.resize(src, self.size, interpolation=cv2.INTER_LINEAR)
        x = src.transpose(2, 0, 1) / 255.0
        x = np.asarray(x[np.newaxis, :, :, :], dtype=np.float32)
        return x

    # ベクトルデータを入力し，指定したレイヤの特徴ベクトルデータを渡す
    def get_feat(self, img_array, layer):
        with chainer.using_config('train', False):
            acts = self.model.extract(img_array, layers=[layer])
        return chainer.cuda.to_cpu(acts[layer].data)

    # 画像を入力し，指定レイヤの特徴ベクトルを１次元化して渡す
    def get_flat_feat(self, img_path, layer):
        x = self.get_img(img_path)
        feat = self.get_feat(x, layer)
        return np.ravel(feat)

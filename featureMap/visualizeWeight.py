# coding:utf-8
# 方針:
# 入力:RGB画像
# 出力:RGB画像・weight画像・RGB画像+weight画像のオーバーレイ画像

# 重み可視化
# https://qiita.com/nagayosi/items/14f243c058f5a1e7044b#layer%E3%81%AE%E3%83%91%E3%83%A9%E3%83%A1%E3%83%BC%E3%82%BF%E3%81%AE%E5%8F%AF%E8%A6%96%E5%8C%96

import os
import math
import glob

import numpy as np
import chainer
from chainer import cuda, serializers
import matplotlib.pyplot as plt

from vgg16 import VGG16

os.environ['PATH'] += ':/usr/local/cuda-8.0/bin:/usr/local/cuda-8.0/bin'
xp = cuda.cupy if cuda.available else np


class VisualizeFeatureMap:
    def __init__(self, model, model_path, resize, gpu=0):
        self.size = resize
        self.model = model  # 入力するモデルは特徴マップが取り出せるようにしておくこおと"
        serializers.load_npz(model_path, model)
        if gpu >= 0:
            chainer.cuda.get_device_from_id(gpu).use()
            model.to_gpu(gpu)
        print("--- Create VisualizeFeatureMap Object ---")

    def get_weight(self, img_path):
        print("INPUT:{}".format(img_path))
        x, t = data.fetch_img_and_label(img_path=img_path, resize=self.size)
        x, t = xp.asarray(x, dtype=xp.float32), xp.asarray(t, dtype=xp.int32)
        return self.model.get_feat(x=x)

    def vis_conv_weight(self, layer_weight, save_name="weight.png"):
        # Variableの入力は許さない
        layer_weight = cuda.to_cpu(layer_weight)  # from cupy to numpy
        batch, ch, h, w = layer_weight.shape
        print(' layer shape --> {}'.format(layer_weight.shape))
        # plt.figure()
        plt.figure(figsize=(15, 15))
        plt.subplots_adjust(left=0.001, right=0.999, top=0.999, bottom=0.001, hspace=0.01, wspace=0.01)
        length_row_and_col = math.ceil(math.sqrt(ch))
        for i in range(ch):
            im = layer_weight[0, i]
            plt.subplot(length_row_and_col, length_row_and_col, i + 1)
            plt.axis('off')
            plt.imshow(im, cmap='jet')
        plt.savefig(save_name)


if __name__ == '__main__':
    model = DANN(class_num=40)
    model_path = "./Result/ue1000_2_real500_cls40/model50000.npz"
    resize = (224, 224)
    gpu = 0
    glob_path = "/home/osumikosuke/MPRG/Data/SIE/item40ALL/class40/patch100/real/*/*.png"
    paths = glob.glob(glob_path)

    visMap = VisualizeFeatureMap(model, model_path, resize, gpu)
    weights = visMap.get_weight(paths[0])
    visMap.vis_conv_weight(weights[2].data)

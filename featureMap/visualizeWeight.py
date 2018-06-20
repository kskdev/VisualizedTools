# coding:utf-8
# 重み可視化
# https://qiita.com/nagayosi/items/14f243c058f5a1e7044b#layer%E3%81%AE%E3%83%91%E3%83%A9%E3%83%A1%E3%83%BC%E3%82%BF%E3%81%AE%E5%8F%AF%E8%A6%96%E5%8C%96

import os
import math

import numpy as np
import cv2
import chainer  # v2
import matplotlib.pyplot as plt

from featureMap import vgg16

os.environ['PATH'] += ':/usr/local/cuda-8.0/bin:/usr/local/cuda-8.0/bin'


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


if __name__ == '__main__':
    image_path = "input.png"
    model_path = "model_file_path.npz"
    target_layer = 'conv1_1'
    output_name = "featureMap.png"
    class_num = 40
    input_size = (224, 224)
    gpu = -1

    fe = FeatureExtractor(class_num, model_path, input_size, gpu)
    img = fe.get_img(image_path)
    feature = fe.get_feat(img_array=img, layer=target_layer)

    batch, ch, h, w = feature.shape
    print(' layer shape : {}'.format(feature.shape))
    plt.figure(figsize=(15, 15))
    plt.subplots_adjust(left=0.001, right=0.999, top=0.999, bottom=0.001, hspace=0.01, wspace=0.01)
    grid_length = math.ceil(math.sqrt(ch))
    for i in range(ch):
        im = feature[0, i]
        plt.subplot(grid_length, grid_length, i + 1)
        plt.axis('off')
        plt.imshow(im, cmap='jet')
    plt.savefig(output_name)

# coding:utf-8

import os
import math

import matplotlib.pyplot as plt

from Common import common, vgg16


if __name__ == '__main__':
    image_path = "../Common/input.jpg"
    model_path = "../Common/VGG16.model"
    target_layer = 'conv5_3'
    output_name = "./featureMap.png"
    class_num = 1000
    input_size = (224, 224)
    gpu = 0

    model = vgg16.VGG16(class_num=class_num)

    fe = common.FeatureExtractor(model, class_num, model_path, input_size, gpu)
    fe.get_top1_label(image_path)
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

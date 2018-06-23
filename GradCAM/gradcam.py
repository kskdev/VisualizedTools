# coding:utf-8

# Referred : https://github.com/tsurumeso/chainer-grad-cam
# anaconda3-4.2.0 (Python3.5) on Ubuntu 14.04
# chainer:2.0.0
# cupy:1.0.0

import argparse
import os
import copy

import cv2
import numpy as np
import chainer
import chainer.functions as F
from chainer import cuda, utils

from Common.common import FeatureExtractor

os.environ["PATH"] = "/usr/local/cuda-8.0/bin:/usr/bin:"


class GuidedReLU(chainer.function.Function):

    def forward(self, x):
        xp = chainer.cuda.get_array_module(x[0])
        self.retain_inputs(())
        self.retain_outputs((0,))
        y = xp.maximum(x[0], 0)
        return y,

    def backward_cpu(self, x, gy):
        y = self.output_data[0]
        return utils.force_array(gy[0] * (y > 0) * (gy[0] > 0)),

    def backward_gpu(self, x, gy):
        y = self.output_data[0]
        gx = cuda.elementwise(
            'T y, T gy', 'T gx',
            'gx = (y > 0 && gy > 0) ? gy : (T)0',
            'relu_bwd')(y, gy[0])
        return gx,


class BaseBackprop(object):
    def __init__(self, model):
        self.model = model
        self.size = model.size
        self.xp = model.xp

    def backward(self, x, label, layer, class_num=1000):
        with chainer.using_config('train', False):
            acts = self.model.extract(x, layers=[layer, 'prob'])
        # data = cuda.to_cpu(acts['prob'].data[0])
        # top5_prob = np.sort(data)[::-1][:5]
        # top5_indx = np.argsort(data)[::-1][:5]
        # for n, (i, p) in enumerate(zip(top5_indx, top5_prob)):
        #     print("TOP{} | ClassID:{:>3} | Prob:{} \n{}".format(n + 1, i, p, "*" * 30))

        one_hot = self.xp.zeros((1, class_num), dtype=np.float32)
        if label == -1:  # -1なら最大スコアのindexを1とするone-hotを作成
            one_hot[:, acts['prob'].data.argmax()] = 1
        else:  # 引数のindexを1とするone-hotを作成
            one_hot[:, label] = 1

        self.model.cleargrads()
        loss = F.sum(chainer.Variable(one_hot) * acts['prob'])
        loss.backward(retain_grad=True)
        return acts


class GradCAM(BaseBackprop):
    def __init__(self, model):
        super(GradCAM, self).__init__(model)

    def generate(self, x, label, layer, class_num):
        acts = self.backward(x, label, layer, class_num)
        weights = self.xp.mean(acts[layer].grad, axis=(2, 3))
        gcam = self.xp.tensordot(weights[0], acts[layer].data[0], axes=(0, 0))
        gcam = (gcam > 0) * gcam / gcam.max()
        gcam = chainer.cuda.to_cpu(gcam * 255)
        gcam = cv2.resize(np.uint8(gcam), (self.size, self.size))
        return gcam


class GuidedBackprop(BaseBackprop):

    def __init__(self, model):
        super(GuidedBackprop, self).__init__(copy.deepcopy(model))
        for key, funcs in self.model.functions.items():
            for i in range(len(funcs)):
                if funcs[i] is F.relu:
                    funcs[i] = GuidedReLU()

    def generate(self, x, label, layer, class_num):
        acts = self.backward(x, label, layer, class_num)
        gbp = chainer.cuda.to_cpu(acts['input'].grad[0])
        gbp = gbp.transpose(1, 2, 0)
        return gbp


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', '-i', default='../Common/input.jpg')
    p.add_argument('--model_path', '-m', default='../Common/VGG16.model')
    p.add_argument('--gpu', '-g', type=int, default=-1)
    p.add_argument('--label', '-y', type=int, default=-1)
    p.add_argument('--class_num', '-c', type=int, default=1000)
    p.add_argument('--layer', '-l', default='conv5_3')
    p.add_argument('--save_name', '-s', default='img_')
    args = p.parse_args()

    from Common.vgg16 import VGG16

    model = VGG16(class_num=args.class_num)
    size = (224, 224)

    fe = FeatureExtractor(model, args.class_num, args.model_path, size, args.gpu)
    raw_img = fe.get_raw_img(args.input)
    input_img = fe.get_img(args.input)

    grad_cam = GradCAM(model)
    guided_bp = GuidedBackprop(model)

    gcam = grad_cam.generate(input_img, args.label, args.layer, args.class_num)
    gbp = guided_bp.generate(input_img, args.label, args.layer, args.class_num)

    # Guided Grad CAM
    ggcam = gbp * gcam[:, :, np.newaxis]
    ggcam -= ggcam.min()
    ggcam = 255 * ggcam / ggcam.max()

    # Grad CAM
    heatmap = cv2.applyColorMap(gcam, cv2.COLORMAP_JET)
    gcam = np.float32(raw_img) + np.float32(heatmap)
    gcam = 255 * gcam / gcam.max()

    cv2.imwrite('{}Source.png'.format(args.save_name), raw_img)
    cv2.imwrite('{}GradCam.png'.format(args.save_name), gcam)
    cv2.imwrite('{}GuidedGradCam.png'.format(args.save_name), ggcam)

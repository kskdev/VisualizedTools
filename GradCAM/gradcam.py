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
        data = cuda.to_cpu(acts['prob'].data[0])
        top5_prob = np.sort(data)[::-1][:5]
        top5_indx = np.argsort(data)[::-1][:5]
        for n, (i, p) in enumerate(zip(top5_indx, top5_prob)):
            print("TOP{} | ClassID:{:>3} | Prob:{}".format(n + 1, i, p))
        print("*" * 30)

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


class VisualizeMap:
    def __init__(self, class_num, gpu=-1):
        # create network object
        from featureMap import vgg16
        self.model = vgg16.VGG16(class_num=class_num)
        print("--- Created network object ---")
        self.class_num = class_num

        if gpu >= 0:
            chainer.cuda.get_device_from_id(gpu).use()
            self.model.to_gpu()

    def get_top1(self, path):
        # Load Image
        src = cv2.imread(path, cv2.IMREAD_COLOR)  # 3ch image
        src = cv2.resize(src, (self.model.size, self.model.size), interpolation=cv2.INTER_LINEAR)
        x = src.transpose(2, 0, 1) / 255.0  # TODO Be careful Normalization method
        x = np.asarray(x[np.newaxis, :, :, :], dtype=np.float32)

        with chainer.using_config('train', False):
            # acts = self.model.extract(x, layers=[layer, 'prob'])
            acts = self.model.extract(x, layers=['prob', 'prob'])
        data = cuda.to_cpu(acts['prob'].data[0])
        top1_prob = np.sort(data)[::-1][:1]
        top1_indx = np.argsort(data)[::-1][:1]
        return top1_indx[0], np.round(top1_prob[0], 4)

    def create_map(self, path, save_name, use_model, label, layer):
        chainer.serializers.load_npz(use_model, self.model)
        print("--- Load network parameter ---")

        grad_cam = GradCAM(self.model)
        guided_bp = GuidedBackprop(self.model)

        # Load Image
        src = cv2.imread(path, cv2.IMREAD_COLOR)  # 3ch image
        src = cv2.resize(src, (self.model.size, self.model.size), interpolation=cv2.INTER_LINEAR)
        x = src.transpose(2, 0, 1) / 255.0  # TODO Be careful Normalization method
        x = np.asarray(x[np.newaxis, :, :, :], dtype=np.float32)

        gcam = grad_cam.generate(x, label, layer, self.class_num)
        gbp = guided_bp.generate(x, label, layer, self.class_num)

        ggcam = gbp * gcam[:, :, np.newaxis]
        ggcam -= ggcam.min()
        ggcam = 255 * ggcam / ggcam.max()

        heatmap = cv2.applyColorMap(gcam, cv2.COLORMAP_JET)
        gcam = np.float32(src) + np.float32(heatmap)
        gcam = 255 * gcam / gcam.max()

        pred, prob = self.get_top1(path)
        pred_info = "visLabel{}_pred{}_Prob{}".format(str(label), str(pred), str(prob))
        cv2.imwrite('{}_{}src.png'.format(pred_info, save_name), src)
        cv2.imwrite('{}_{}_gcam.png'.format(pred_info, save_name), gcam)
        cv2.imwrite('{}_{}_ggcam.png'.format(pred_info, save_name), ggcam)
        print("--- Save Result ---")


if __name__ = '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', '-i', default='./dog_cat.png')
    p.add_argument('--model', '-m', default='trainedModel.npz')
    p.add_argument('--gpu', '-g', type=int, default=-1)
    p.add_argument('--label', '-y', type=int, default=-1)
    p.add_argument('--class_num', '-c', type=int, default=1000)
    p.add_argument('--layer', '-l', default='conv5_3')
    p.add_argument('--save', '-s', default='img')
    p.add_argument('--prefix', '-p', default='png')
    args = p.parse_args()
    
    save_name = "{}.{}".format(args.save, args.prefix)
    vm = VisualizeMap(args.class_num, args.gpu):
    vm.create_map(args.input, save_name, args.model, args.label, args.layer):



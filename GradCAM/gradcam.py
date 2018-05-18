# coding:utf-8

# Visualized Attention
# Referred : https://github.com/tsurumeso/chainer-grad-cam

import argparse
import copy
import os

import cv2
import numpy as np
import chainer
import chainer.functions as F
from chainer import cuda
from chainer import utils
from chainer.dataset import download
from chainer.serializers import npz


# --- Backpop Class ---
class BaseBackprop(object):

    def __init__(self, model):
        self.model = model
        self.size = model.size
        self.xp = model.xp

    def backward(self, x, label, layer, class_num=1000):
        with chainer.using_config('train', False):
            acts = self.model.extract(x, layers=[layer, 'prob'])

        one_hot = self.xp.zeros((1, class_num), dtype=np.float32)
        if label == -1: one_hot[:, acts['prob'].data.argmax()] = 1
        else: one_hot[:, label] = 1

        self.model.cleargrads()
        loss = F.sum(chainer.Variable(one_hot) * acts['prob'])
        loss.backward(retain_grad=True)
        return acts



# --- GradCAM Class ---
class GradCAM(BaseBackprop):
    def __init__(self, model, class_num=1000):
        super(GradCAM, self).__init__(model)

    def generate(self, x, label, layer, class_num):
        acts = self.backward(x, label, layer, class_num)
        weights = self.xp.mean(acts[layer].grad, axis=(2, 3))
        gcam = self.xp.tensordot(weights[0], acts[layer].data[0], axes=(0, 0))
        gcam = (gcam > 0) * gcam / gcam.max()
        gcam = chainer.cuda.to_cpu(gcam * 255)
        gcam = cv2.resize(np.uint8(gcam), (self.size, self.size))
        return gcam



# --- GuidedBackprop Class ---
class GuidedBackprop(BaseBackprop):
    def __init__(self, model, class_num=1000):
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



# --- GuidedReLU Class ---
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


def convert_caffemodel_to_npz(cls, path_caffemodel, path_npz):
    from chainer.links.caffe.caffe_function import CaffeFunction
    caffemodel = CaffeFunction(path_caffemodel)
    npz.save_npz(path_npz, caffemodel, compression=False)

def _make_npz(path_npz, url, model):
    path_caffemodel = download.cached_download(url)
    print('Now loading caffemodel (usually it may take few minutes)')
    convert_caffemodel_to_npz(path_caffemodel, path_npz)
    npz.load_npz(path_npz, model)
    return model

def _retrieve(name, url, model):
    root = download.get_dataset_directory('pfnet/chainer/models/')
    path = os.path.join(root, name)
    return download.cache_or_load_file(
        path, lambda path: _make_npz(path, url, model),
        lambda path: npz.load_npz(path, model))





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

def create_gradcam_map():
    # create network object
    import vgg
    model = vgg.VGG16(class_num=args.class_num)
    print("--- Created network object ---")
    chainer.serializers.load_npz(args.model, model)
    print("--- Load network parameter ---")
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    # Create Grad Cam Object
    grad_cam = GradCAM(model)
    guided_backprop = GuidedBackprop(model)

    # Load Image
    # src = cv2.imread(args.input, 1)
    # src = cv2.resize(src, (model.size, model.size))
    # x = src.astype(np.float32) - np.float32([103.939, 116.779, 123.68])
    # x = x.transpose(2, 0, 1)[np.newaxis, :, :, :]

    # Load Image
    src = cv2.imread(args.input, cv2.IMREAD_COLOR)  # 3ch image
    src = cv2.resize(src, (model.size, model.size), interpolation=cv2.INTER_LINEAR)
    x = src.transpose(2, 0, 1) / 255.0
    x = np.asarray(x[np.newaxis, :, :, :], dtype=np.float32)

    # Generate Map
    gcam = grad_cam.generate(x, args.label, args.layer, args.class_num)
    gbp = guided_backprop.generate(x, args.label, args.layer, args.class_num)

    # ggcam = gbp * gcam[:, :, np.newaxis]
    # ggcam -= ggcam.min()
    # ggcam = 255 * ggcam / ggcam.max()
    # cv2.imwrite('{}_ggcam.{}'.format(args.save, args.prefix), ggcam)

    # gbp -= gbp.min()
    # gbp = 255 * gbp / gbp.max()
    # cv2.imwrite('{}_gbp.{}'.format(args.save, args.prefix), gbp)

    heatmap = cv2.applyColorMap(gcam, cv2.COLORMAP_JET)
    gcam = np.float32(src) + np.float32(heatmap)
    gcam = 255 * gcam / gcam.max()
    cv2.imwrite('{}_gcam.{}'.format(args.save, args.prefix), gcam)
    print("--- Save Result ---")


if __name__ == '__main__':
    create_gradcam_map()

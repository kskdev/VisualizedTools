# -*- coding: utf-8 -*-
# python 3.5.2 with Anaconda3 4.2.0

import glob

import bhtsne
import numpy as np
from PIL import Image

from common import flatten, hex_to_rgb, c_cycle, FeatureExtractor


def compress_from_path(lst, class_num, model_path, size, layer, seed=-1):
    # convert path list from 2dim to 1dim
    flatten_lst = flatten(lst)

    # create network object
    fe = FeatureExtractor(class_num, model_path, size, gpu)

    # get feature list
    vec = np.asarray(list(map(lambda f: fe.get_flat_feat(f, layer), flatten_lst)))

    # Convert image from 3dim(RGB) to 1dim (for processing tSNE)
    vec = np.asarray(list(map(lambda v: np.ravel(v), vec)), np.float64)

    # Run tSNE (Dimensional compression)
    compressed_vec = bhtsne.tsne(vec, rand_seed=seed)

    # Get split position
    len_list = [len(n) for n in lst]
    split_pos = [sum(len_list[:n + 1]) for n in range(len(len_list))]
    split_pos = split_pos[:len(len_list) - 1]

    # Return split vector
    return np.split(compressed_vec, split_pos, axis=0)


if __name__ == '__main__':
    # Get image path list
    img_list = [
        glob.glob("./Data1/*.png"),
        glob.glob("./Data2/*.png"),
        glob.glob("./Data3/*.png")
    ]

    # Network Setting
    class_num = 40
    model_path = "chainer_model_file.npz"
    gpu = 0
    layer = "fc8"
    size = (224, 224)
    save_name = "map.png"

    # Generating graph property (It is not necessary to change the settings)
    canvas_size = (16000, 10000)
    canvas_color = "#FFFFFF"
    paste_size = (200, 200)  # Resizing image to paste canvas
    paste_scale = 0.96  # Adjust image that does not extend beyond the canvas
    frame_width = 10

    # **********************************
    # Create canvas
    canvas = Image.new('RGB', canvas_size, hex_to_rgb(canvas_color))

    # Compressing dimension by tSNE
    coordinate = compress_from_path(img_list, class_num, model_path, size, layer)

    # Get minimum and maximum in compressed data
    minimum, maximum = np.vstack(coordinate).min(), np.vstack(coordinate).max()

    # Process for each data
    for i, (paths, compressed_xy) in enumerate(zip(img_list, coordinate)):

        # Normalization in min-max
        norm_xy = (compressed_xy - minimum) / (maximum - minimum)
        # Set RGB color from HTML color
        frame_color = hex_to_rgb(c_cycle[i])

        # Paste image on canvas(Process for each file)
        for j, p in enumerate(paths):
            # Decide pasting image position
            x = int(norm_xy[j][0] * canvas_size[0] * paste_scale)
            y = int(norm_xy[j][1] * canvas_size[1] * paste_scale)

            # Load image (and resize)
            img = Image.open(p).resize(paste_size)

            # Paste an image on a single color bg (Use single color bg as a frame)
            lt = int(img.size[0] / frame_width), int(img.size[1] / frame_width)
            rb = int(img.size[0]) + lt[0] * 2, int(img.size[1]) + lt[1] * 2
            bg = Image.new('RGB', rb, frame_color)
            bg.paste(img, lt)

            # Paste image with frame on canvas
            canvas.paste(bg, (x, y))

    canvas.save(save_name, quality=100)

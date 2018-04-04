# -*- coding: utf-8 -*-
# python 3.5.2 with Anaconda3 4.2.0
# coded by osmksk05

# Clustering and Visualization by tSNE

import glob
import numpy as np
import bhtsne
from PIL import Image


def flatten(nested_list):
    return [e for inner_list in nested_list for e in inner_list]


def compress_from_path(lst, resize=(64, 64), seed=-1):
    # convert path list from 2dim to 1dim
    flatten_lst = flatten(lst)

    # Load image each paths
    img_vec = list(map(lambda f: Image.open(f).resize(resize), flatten_lst))

    # Convert image from 3dim(RGB) to 1dim (for processing tSNE)
    img_vec = np.asarray(list(map(lambda v: np.ravel(v), img_vec)), np.float64)

    # Run tSNE (Dimensional compression)
    compressed_vec = bhtsne.tsne(img_vec, rand_seed=seed)

    # Get split position
    len_list = [len(n) for n in lst]
    split_pos = [sum(len_list[:n + 1]) for n in range(len(len_list))]
    split_pos = split_pos[:len(len_list) - 1]

    # Return split vector each domain
    return np.split(compressed_vec, split_pos, axis=0)


def hex_to_rgb(hex_code):
    hex_code = hex_code.lstrip('#')
    lv = len(hex_code)
    return tuple(int(hex_code[ii:ii + lv // 3], 16) for ii in range(0, lv, lv // 3))


if __name__ == '__main__':

    # Get image path list
    p1 = glob.glob("./Data1/*.jpg")
    p2 = glob.glob("./Data2/*.jpg")
    p3 = glob.glob("./Data3/*.jpg")

    # Set image path list
    img_list = [p1, p2, p3]

    # Set save file name
    save_name = "mapped_images.png"

    # Define color pattern
    c_cycle = ("#E51400", "#1BA1E2", "#339933", "#A200FF", "#E671B8", "#F09609", "#515151", "#A05000")

    canvas_size = (24000, 20000)
    canvas_color = "#333333"

    resize_for_tsne = (128, 128)  # Input for compressing target image
    paste_size = (200, 200)  # Resizing image to paste canvas
    paste_scale = 0.96  # Adjust image that does not extend beyond the canvas
    frame_width = 10

    # **********************************
    # Create canvas
    canvas = Image.new('RGB', canvas_size, hex_to_rgb(canvas_color))

    # Compressing dimension by tSNE
    coordinate = compress_from_path(img_list, resize_for_tsne, seed=-1)

    # Get minimum and maximum in compressed data
    minimum, maximum = np.vstack(coordinate).min(), np.vstack(coordinate).max()

    # Process each data
    for i, (paths, compressed_xy) in enumerate(zip(img_list, coordinate)):

        # Normalization in min-max
        norm_xy = (compressed_xy - minimum) / (maximum - minimum)
        # Set RGB color from HTML color
        frame_color = hex_to_rgb(c_cycle[i])

        # Paste image on canvas(process every path)
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

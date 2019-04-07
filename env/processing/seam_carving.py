"""
Usage: python carver.py <r/c> <scale> <image_in> <image_out>
Copyright 2018 Karthik Karanth, MIT License
"""


import sys
from numba import jit
import numpy as np
from imageio import imread, imwrite
from scipy.ndimage.filters import convolve


from PIL.Image import open
from os.path import join
from clean_csv import create_dir #processing
import os

# tqdm is not strictly necessary, but it gives us a pretty progress bar
# to visualize progress.
from tqdm import trange

def calc_energy(img):
    filter_du = np.array([
        [1.0, 2.0, 1.0],
        [0.0, 0.0, 0.0],
        [-1.0, -2.0, -1.0],
    ])
    # This converts it from a 2D filter to a 3D filter, replicating the same
    # filter for each channel: R, G, B
    filter_du = np.stack([filter_du] * 3, axis=2)

    filter_dv = np.array([
        [1.0, 0.0, -1.0],
        [2.0, 0.0, -2.0],
        [1.0, 0.0, -1.0],
    ])
    # This converts it from a 2D filter to a 3D filter, replicating the same
    # filter for each channel: R, G, B
    filter_dv = np.stack([filter_dv] * 3, axis=2)

    img = img.astype('float32')
    convolved = np.absolute(convolve(img, filter_du)) + np.absolute(convolve(img, filter_dv))

    # We sum the energies in the red, green, and blue channels
    energy_map = convolved.sum(axis=2)

    return energy_map

@jit
def minimum_seam(img):
    r, c, _ = img.shape
    energy_map = calc_energy(img)

    M = energy_map.copy()
    backtrack = np.zeros_like(M, dtype=np.int)

    for i in range(1, r):
        for j in range(0, c):
            # Handle the left edge of the image, to ensure we don't index -1
            if j == 0:
                idx = np.argmin(M[i - 1, j:j + 2])
                backtrack[i, j] = idx + j
                min_energy = M[i - 1, idx + j]
            else:
                idx = np.argmin(M[i - 1, j - 1:j + 2])
                backtrack[i, j] = idx + j - 1
                min_energy = M[i - 1, idx + j - 1]

            M[i, j] += min_energy

    return M, backtrack

@jit
def carve_column(img):
    r, c, _ = img.shape

    M, backtrack = minimum_seam(img)

    # Create a (r, c) matrix filled with the value True
    # We'll be removing all pixels from the image which
    # have False later
    mask = np.ones((r, c), dtype=np.bool)

    # Find the position of the smallest element in the
    # last row of M
    j = np.argmin(M[-1])

    for i in reversed(range(r)):
        # Mark the pixels for deletion
        mask[i, j] = False
        j = backtrack[i, j]

    # Since the image has 3 channels, we convert our
    # mask to 3D
    mask = np.stack([mask] * 3, axis=2)

    # Delete all the pixels marked False in the mask,
    # and resize it to the new image dimensions
    img = img[mask].reshape((r, c - 1, 3))

    return img

def crop_c(img, scale_c):
    r, c, _ = img.shape
    new_c = int(scale_c * c)

    for i in trange(c - new_c): # use range if you don't want to use tqdm
        img = carve_column(img)

    return img

def crop_r(img, scale_r):
    img = np.rot90(img, 1, (0, 1))
    img = crop_c(img, scale_r)
    img = np.rot90(img, 3, (0, 1))
    return img

def find_scale(image_path):
    img = imread(image_path)
    h, w, _ = img.shape
    print("w: ", str(w), ", h: ", str(h))
    return 224./w, 224./h

# identify the scale carve and whether to resize, return the scale option and resize option as tuple
# example return (b/r/c/None, h/w/None/b)
def choose_which(image_path):
    w_s, h_s = find_scale(image_path)
    print("w ratio: ", str(w_s), " , h ratio: ", str(h_s))
    if w_s < 1:
        if h_s < 1:
            return 'b', None
        elif h_s > 1:
            return 'c', 'h'
        else:
            return 'c', None
    elif w_s > 1:
        if h_s < 1:
            return 'r', 'w'
        elif h_s > 1:
            return None, 'b'
        else:
            return None, 'w'
    else:
        if h_s < 1:
            return 'r', None
        elif h_s > 1:
            return None, 'h'
        else:
            return None, None

def carve(in_filename, out_filename):
    img = imread(in_filename)
    h, w, _ = img.shape
    print("inside carve, h: ", str(h), " , w: ", str(w))
    which_axis, resize = choose_which(in_filename)
    print("which axis: ", which_axis, " , resize: ", resize)
    # im = open(in_filename)
    # w, h = im.size
    r = 224. / h
    c = 224. / w
    print("resulting size: ", str(c * w), " , ", str(r * h))
    if which_axis == 'r':
        scale = r
        out = crop_r(img, scale)
    elif which_axis == 'c':
        scale = c
        out = crop_c(img, scale)
    elif which_axis == 'b':
        scale = c
        out1 = crop_c(img, scale)
        scale1 = r
        out = crop_r(out1, scale1)
    else:
        pass
        # print('usage: seam_carving.py <r/c> <scale> <image_in> <image_out>', file=sys.stderr)
        # sys.exit(1)
    if resize == 'h':
        final = out.resize((out.shape[0], 224))
    elif resize == 'w':
        final = out.resize((224, out.shape[1]))
    elif resize == 'b':
        final = out.resize((224, 224))
    else:
        final = out
    print("shape of final: ", str(final.shape))

    imwrite(out_filename, final)

def main():
    if len(sys.argv) != 3:
        #print('usage: seam_carving.py <r/c> <scale> <image_in> <image_out>', file=sys.stderr)
        print('usage: seam_carving.py <dir_image_in> <dir_image_out>', file=sys.stderr)
        sys.exit(1)

    #which_axis = sys.argv[1]
    #scale = float(sys.argv[2])
    #in_filename = sys.argv[1]#[3]
    #out_filename = sys.argv[2]#[4]

    in_dir = sys.argv[1]
    out_dir = sys.argv[2]
    create_dir(out_dir)


    for img_name in os.listdir(in_dir):
        img_path = join(in_dir, img_name)
        out_path = join(out_dir, img_name)
        carve(img_path, out_path)

if __name__ == '__main__':
    main()

# sample run of code:
# python3 seam_carving r 0.5 image.jpg cropped.jpg
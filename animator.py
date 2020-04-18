import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import glob
import time
import argparse
from timelapse import *

def find_indices(bgs, bgs_many):
    result = np.zeros(len(bgs), dtype=int)

    index = 0
    for i in range(len(bgs_many)):
        if np.array_equal(bgs[index], bgs_many[i]):
            result[index] = i
            index += 1
            if index >= len(bgs):
                break
    
    return result

def make_frames(plate, bgs, bgs_many, mapping, ppf, interval, many=False, inorder=True):
    count = len(bgs)
    if many:
        count = len(bgs_many)
    
    folder_name = 'frame_{}_{}_{}_{}_{}_{}'.format(mapping, count, ppf, interval, many, inorder)
    if os.path.isdir('./{}/{}'.format(name, folder_name)):
        return folder_name, count

    if many:
        mask = bin_map_mask(ppf, plate, inorder)
    else:
        mask = bin_map_mask(len(bgs), plate, inorder)

    os.makedirs('{}/{}'.format(name, folder_name), exist_ok=True)

    print(many)
    if many:
        indices = np.linspace(0, interval * (ppf-1), ppf, dtype=int)
        print(indices)
        for i in range(count):
            images = bgs_many[indices]
            img = apply_masks(images, mask)
            cv2.imwrite('{}/{}/{:03d}.jpg'.format(name, folder_name, i), img)
            indices = (indices + 1) % count

    else:
        for i in range(count):
            img = apply_masks(bgs, mask, i)
            cv2.imwrite('{}/{}/{:03d}.jpg'.format(name, folder_name, i), img)

    return folder_name, count

def make_gif(name, folder_name):
    input_path = '{}/{}/*.jpg'.format(name, folder_name)
    output_path = '{}/animated_{}_{}.gif'.format(name, name, folder_name[6:])
    cmd = 'magick convert -delay 1 -resize 1500x1000 -loop 0 {} {}'.format(input_path, output_path)
    os.system(cmd)

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("name", type=str)
    parser.add_argument("mapping", type=str)
    parser.add_argument("--inorder", type=str2bool, default=True)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--interval", type=int, default=0)
    parser.add_argument("--many", type=str2bool,  default=True)
    opt = parser.parse_args()

    name = opt.name
    mapping = opt.mapping
    img_inorder = opt.inorder
    anim_inorder = True
    many = opt.many
    interval = opt.interval
    count = opt.count
    
    print(many, img_inorder)
    face, bgs, bgs_many = read_images(name, many=many)
    
    start = time.time()
    folder_name, count = make_frames(face, bgs, bgs_many, mapping, ppf=count, interval=interval, many=many, inorder=img_inorder)
    print("done making frames")
    make_gif(name, folder_name)

    end = time.time()
    print(end - start)

    

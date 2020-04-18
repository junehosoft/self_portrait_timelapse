import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import glob
import time
import os

# read the images
def read_images(name, many=False):
    plate_files = glob.glob('{}/plate/*.jpg'.format(name))
    plates = []
    for p in plate_files:
        plates.append(cv2.imread(p))
    plates = np.array(plates)
    plate = plates[0]

    bgs = []
    bg_files = glob.glob('{}/bg/*.jpg'.format(name))
    for bg in bg_files:
        bgs.append(cv2.imread(bg))
    bgs = np.array(bgs)

    bgs_many = []
    if many:
        bg_many_files = glob.glob('{}/bg_many/*.jpg'.format(name))
        for bg in bg_many_files:
            bgs_many.append(cv2.imread(bg))
    bgs_many = np.array(bgs_many)

    return plate, bgs, bgs_many

# bins evenly the luminance of face 
def bin_map(backgrounds, face, inorder=True):
    face_bw = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    min_lum = np.amin(face_bw)
    max_lum = np.amax(face_bw)
    diff = max_lum - min_lum
    result = np.zeros((face.shape), np.uint8)
    for i in range(face.shape[0]):
        for j in range(face.shape[1]):
            lum = face_bw[i,j]
            bin_lum = int((lum - min_lum) * len(backgrounds) / diff)
            if (bin_lum >= len(backgrounds)):
                bin_lum = len(backgrounds) - 1
            if not inorder:
                bin_lum = len(backgrounds) - 1 - bin_lum
            result[i, j] = backgrounds[bin_lum][i, j]
    
    return result

def bin_map_mask(count, face, inorder=True):
    face_bw = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    min_lum = np.amin(face_bw)
    max_lum = np.amax(face_bw)
    diff = max_lum - min_lum
    result = np.zeros((count, face.shape[0], face.shape[1]), np.uint8)
    for i in range(face.shape[0]):
        for j in range(face.shape[1]):
            lum = face_bw[i,j]
            bin_lum = int((lum - min_lum) * count / diff)
            if (bin_lum >= count):
                bin_lum = count - 1
            if not inorder:
                bin_lum = count - 1 - bin_lum
            result[bin_lum, i, j] = 1
    
    return result

# chooses pixels based on luminance closenesss
def lum_map(backgrounds, face):
    face_bw = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY).astype(np.int16)
    backgrounds_bw = np.zeros(backgrounds.shape[:3])
    for i in range(len(backgrounds_bw)):

        backgrounds_bw[i] = cv2.cvtColor(backgrounds[i], cv2.COLOR_BGR2GRAY).astype(np.int16)

    result = np.zeros((face.shape), np.uint8)
    diff = np.zeros((len(backgrounds_bw), face_bw.shape[0], face_bw.shape[1]))
    #print('face range', np.min(face_bw), np.max(face_bw))
    for i in range(len(backgrounds_bw)):
        diff[i] = np.absolute(face_bw - backgrounds_bw[i])

    #print(result.shape)
    for i in range(face.shape[0]):
        for j in range(face.shape[1]):
            min_index = -1
            min_diff = 300
            for k in range(len(backgrounds_bw)):
                if diff[k, i, j] < min_diff:
                    min_diff = diff[k,i,j]
                    min_index = k
            if min_index == -1:
                print("wtf")
            result[i, j] = backgrounds[min_index][i,j]
    
    return result

# chooses pixels based on luminance closenesss
def lum_map_mask(backgrounds, face):
    face_bw = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY).astype(np.int16)
    backgrounds_bw = np.zeros(backgrounds.shape[:3])

    for i in range(len(backgrounds_bw)):
        backgrounds_bw[i] = cv2.cvtColor(backgrounds[i], cv2.COLOR_BGR2GRAY).astype(np.int16)
    result = np.zeros((len(backgrounds), face.shape[0], face.shape[1]), np.uint8)
    diff = np.zeros((len(backgrounds_bw), face_bw.shape[0], face_bw.shape[1]))
    for i in range(len(backgrounds_bw)):
        diff[i] = np.absolute(face_bw - backgrounds_bw[i])
    for i in range(face.shape[0]):
        for j in range(face.shape[1]):
            min_index = -1
            min_diff = 300
            for k in range(len(backgrounds_bw)):
                if diff[k, i, j] < min_diff:
                    min_diff = diff[k,i,j]
                    min_index = k
            if min_index == -1:
                print("wtf")
            result[min_index, i, j] = 1
    
    return result

def apply_masks(backgrounds, mask, offset=0):
    assert len(backgrounds) == mask.shape[0]

    result = np.zeros(backgrounds[0].shape, np.uint8)
    for i in range(len(backgrounds)):
        index = (i + offset) % len(backgrounds)
        result += cv2.bitwise_and(backgrounds[index], backgrounds[index], mask=mask[i])

    return result

def layer_masks(backgrounds, mask, inorder):
    N = len(mask)
    bw = np.ones((N, backgrounds[0].shape[0], backgrounds[0].shape[1], 4), np.uint8)
    color = np.zeros((N, backgrounds[0].shape[0], backgrounds[0].shape[1], 4), np.uint8)
    for i in range(N):
        bga = cv2.cvtColor(backgrounds[i], cv2.COLOR_BGR2BGRA)
        color[i] = cv2.bitwise_and(bga, bga, mask=mask[i])

    gradient = np.linspace(0, 255, N, dtype=np.uint8)
    if not inorder:
        gradient = gradient[::-1]

    for i in range(N):
        bw[i,:,:,:3] *= gradient[i]
        bw[i, :, :, 3] = mask[i] * 255

    return color, bw

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
    parser.add_argument("--tag", type=str, default="")
    opt = parser.parse_args()

    name = opt.name
    inorder = opt.inorder
    mapping = opt.mapping
    tag = opt.tag
    
    face, backgrounds, _ = read_images(name)
    start = time.time()
    if mapping == 'bin':
        masks = bin_map_mask(len(backgrounds), face, inorder)
    elif mapping == 'lum':
        masks = lum_map_mask(backgrounds, face)
    end = time.time()
    print("Time: {:.2f}".format(end - start))
    result = apply_masks(backgrounds, masks)
    #plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    #plt.show()
    color, bw = layer_masks(backgrounds, masks, inorder)

    cv2.imwrite('{}/{}_{}_{}_{}.jpg'.format(name, name, mapping, len(backgrounds), tag), result)
    os.makedirs('{}/layer_color_{}'.format(name, mapping), exist_ok=True)
    os.makedirs('{}/layer_bw_{}'.format(name, mapping), exist_ok=True)
    for i in range(len(masks)):
        cv2.imwrite('{}/layer_color_{}/{:03d}.png'.format(name, mapping, i), color[i])
        cv2.imwrite('{}/layer_bw_{}/{:03d}.png'.format(name, mapping, i), bw[i])


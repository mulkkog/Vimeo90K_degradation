import os
import sys
import cv2
import numpy as np
import os.path as osp
import glob


def generate_LR_Vimeo90K():
    # set parameters
    up_scale = 4
    sigma = 1.6

    # set data dir
    sourcedir = '/home/ubuntu/jijang/Dataset/Vimeo-90K/vimeo_septuplet/sequences/'
    saveLRpath = '/home/ubuntu/jijang/Dataset/Vimeo-90K/BDx4/'
    txt_file = '/home/ubuntu/jijang/Dataset/Vimeo-90K/vimeo_septuplet/sep_trainlist.txt'

    # read all the image paths to a list
    print('Reading image path list ...')
    with open(txt_file) as f:
        train_l = f.readlines()
        train_l = [v.strip() for v in train_l]
    all_img_list = []
    for line in train_l:
        folder = line.split('/')[0]
        sub_folder = line.split('/')[1]
        all_img_list.extend(glob.glob(osp.join(sourcedir, folder, sub_folder, '*')))
    all_img_list = sorted(all_img_list)
    num_files = len(all_img_list)

    # prepare data with augementation
    for i in range(num_files):
        filename = all_img_list[i]
        print('No.{} -- Processing {}'.format(i, filename))

        # read image
        image = cv2.imread(filename)

        # Gaussian filter
        blurred_image = cv2.GaussianBlur(image, (0, 0), sigma)

        # subsampling
        image_LR = blurred_image[::up_scale, ::up_scale]

        folder = filename.split('/')[8]
        sub_folder = filename.split('/')[9]
        name = filename.split('/')[10]

        if not os.path.isdir(osp.join(saveLRpath, folder)):
            os.mkdir(osp.join(saveLRpath, folder))

        if not os.path.isdir(osp.join(saveLRpath, folder, sub_folder)):
            os.mkdir(osp.join(saveLRpath, folder, sub_folder))

        cv2.imwrite(osp.join(saveLRpath, folder, sub_folder, name), image_LR)

    print('Finish LR generation')


if __name__ == "__main__":
    generate_LR_Vimeo90K()

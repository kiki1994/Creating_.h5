#!/usr/bin/env python
# -*-coding: utf-8 -*-

import h5py
import os
import cv2
import math
import numpy as np
import random
import re

def creat_h5(txt_PATH):
    print(txt_PATH)
    seg = txt_PATH.split('/')[:]
    print(seg)
    person_id = seg[-3]
    day_id = seg[-2]
#    print(day_id)
    person_num = int(person_id[1:])
#    print(person_num)
    day_num = int(day_id[3:])
#    print(day_num)
    eye_direction = seg[-1].split('_')[0]
    root_path = './Original_eye/{a}/{b}/{c}'.format(a=person_id,b=day_id,c=eye_direction)
    two_eye_path = './MPIIGaze/Data/Original/p{:2}/day{:02}'.format(person_num,day_num)
    print(two_eye_path)
    head_pose_txt = './headpose_2D/{a}/{b}/{c}_headpose.txt'.format(a=person_id, b=day_id, c=eye_direction)
    save_dir = "./h5_2eyes/{a}/{b}/{c}".format(a=person_id, b=day_id, c=eye_direction)
    os.makedirs(save_dir, exist_ok=True)
    with open(txt_PATH, 'r') as f:
        lines = f.readlines()


    with open(head_pose_txt, 'r') as hf:
        h_lines = hf.readlines()

    num = len(lines)
#    print(num)
    random.shuffle(lines)
    # imgAccu = 0
    imgs = np.zeros([num, 3, 36, 60])
    imgs_2eye = np.zeros([num, 3, 720, 1280])
    labels = np.zeros([num, 4])
    for i in range(num):
        line = lines[i]
        h_line = h_lines[i]
        segments = re.split('\s+', line)[:-1]
        h_segments = re.split('\s+', h_line)[:-1]
#        print(segments[0] == h_segments[0])                #same nnum?

        img = cv2.imread(os.path.join(root_path, segments[1]))
        img2eye = cv2.imread(os.path.join(two_eye_path, segments[1]))
        img = img.transpose(2, 0, 1)
        img2eye = img2eye.transpose(2, 0, 1)
        imgs[i, :, :, :] = img.astype(np.float32)
        imgs_2eye[i, :, :, :] = img2eye.astype(np.float32)
        labels[i, :] = np.array(list(map(float, [segments[2], segments[3], h_segments[2], h_segments[3]])),
                                dtype="float32")                                                             #gaze and head
#        print(labels[i])


    batchsize = 5
    batchNum = int(math.ceil(1.0 * num / batchsize))

    #imgsMean = np.mean(imgs, axis=0)

    #labelsMean = np.mean(labels, axis=0)
    #labels = (labels - labelsMean)/3


    train_txt_path = os.path.join(save_dir, "trainlist.txt")
    test_txt_path = os.path.join(save_dir, "testlist.txt")
    if os.path.exists(train_txt_path):
        os.remove(train_txt_path)
    if os.path.exists(test_txt_path):
        os.remove(test_txt_path)
    comp_kwargs = {'compression': 'gzip', 'compression_opts': 1}
    for i in range(batchNum):
        start = i * batchsize
        end = min((i + 1) * batchsize, num)
        if i < (batchNum * 3/4):
            filename = os.path.join(save_dir, 'train_{0:07}.h5'.format(i))
        else:
            filename = os.path.join(save_dir, 'test_{0:07}.h5'.format(-(i - batchNum + 1)))
        dir_name = os.path.dirname(os.path.abspath(filename))
        if os.path.exists(dir_name):
            print(filename)
        with h5py.File(filename, 'w') as f:
            f.create_dataset('data1', data=np.array(imgs[start:end]).astype(np.float32), **comp_kwargs)
            f.create_dataset('data2', data=np.array(imgs_2eye[start:end]).astype(np.float32), **comp_kwargs)
            f.create_dataset('labels', data=np.array(labels[start:end]).astype(np.float32), **comp_kwargs)
            if i < (batchNum * 3/4):
                with open(os.path.join(os.path.abspath(save_dir), 'trainlist.txt'), 'a') as fi:
                    fi.write(os.path.join(os.path.abspath(save_dir), 'train_{0:07}.h5').format(i) + '\n')
            else:
                with open(os.path.join(os.path.abspath(save_dir), 'testlist.txt'), 'a') as fi:
                    fi.write(os.path.join(os.path.abspath(save_dir), 'test_{0:07}.h5').format(-(i - batchNum + 1)) + '\n')
    #imgsMean = np.mean(imgs, axis=(1, 2))
    #with open('mean.txt', 'w')as f:
    #    f.write(str(imgsMean[0]) + '\n' + str(imgsMean[1]) + '\n' + str(imgsMean[2]))

# txtPath = r"D:/gaze_detection/gaze_2D/P0/day1/left_gaze.txt"
# creat_h5(txtPath)

root = '/disks/disk0/linyuqi/dataset/data_gaze/gaze_detection/gaze_2D/'
#root ='F:/datebase/zhang_xucong_paper/MPIIGaze/MPIIGaze/Data/gaze_2D/'
for person in [root + a + '/' for a in os.listdir(root)]:
    for day in [person + a + '/' for a in os.listdir(person)]:
        for eye_txt in [day + a + '/' for a in os.listdir(day)]:
            print(eye_txt)
            creat_h5(os.path.abspath(eye_txt))

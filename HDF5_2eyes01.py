#!/usr/bin/env python
# -*-coding: utf-8 -*-
#!/usr/bin/env python
# -*-coding: utf-8 -*-

import h5py
import os
import cv2
import math
import numpy as np
import random
import re

def creat_h5(eye_txt):
    for txt_PATH in [eye_txt + a + '' for a in os.listdir(eye_txt)]:
#        print(txt_PATH)
        seg = txt_PATH.split('/')[:]
#        print(seg)
        person_id = seg[-3]
        day_id = seg[-2]
#        print(person_id)
        person_num = int(person_id[1:])
#        print(person_num)
        day_num = int(day_id[3:])
#        print(day_num)
        eye_direction = seg[-1].split('_')[0]
        save_dir = "./h5_2eyes/{a}/{b}".format(a=person_id, b=day_id)
        if eye_direction == 'left':
            gaze_txt_path_l = './gaze_2D/{a}/{b}/{c}_gaze.txt'.format(a=person_id,b=day_id,c=eye_direction)
            root_path_l = './Original_eye/{a}/{b}/{c}'.format(a=person_id,b=day_id,c=eye_direction)
            two_eye_path = './MPIIGaze/Data/Original/p{:02}/day{:02}'.format(person_num,day_num)
#            print(two_eye_path)
            head_pose_txt_l = './headpose_2D/{a}/{b}/{c}_headpose.txt'.format(a=person_id, b=day_id, c=eye_direction)
            os.makedirs(save_dir, exist_ok=True)
        else:
            gaze_txt_path_r = './gaze_2D/{a}/{b}/{c}_gaze.txt'.format(a=person_id,b=day_id,c=eye_direction)
            root_path_r = './Original_eye/{a}/{b}/{c}'.format(a=person_id, b=day_id, c=eye_direction)
            two_eye_path = './MPIIGaze/Data/Original/p{:2}/day{:02}'.format(person_num, day_num)
#            print(two_eye_path)
            head_pose_txt_r = './headpose_2D/{a}/{b}/{c}_headpose.txt'.format(a=person_id, b=day_id, c=eye_direction)
            os.makedirs(save_dir, exist_ok=True)
            
    print(head_pose_txt_l,root_path_l)
    print(head_pose_txt_r,root_path_r)
    
    with open(gaze_txt_path_l, 'r') as f:
        lines_l = f.readlines()
    with open(gaze_txt_path_r, 'r') as f:
        lines_r = f.readlines()
    with open(head_pose_txt_l, 'r') as hf:
        h_lines_l = hf.readlines()
    with open(head_pose_txt_r, 'r') as hf:
        h_lines_r = hf.readlines()
        
    num = len(lines_l)
#    print(num)
#    random.shuffle(lines_l)
    # imgAccu = 0
    imgs_l = np.zeros([num, 3, 36, 60])
    imgs_r = np.zeros([num, 3, 36, 60])
    imgs_2eye = np.zeros([num, 3, 720, 1280])
    labels = np.zeros([num, 8])
    for i in range(num):
        line_l = lines_l[i]
        line_r = lines_r[i]
        h_line_l = h_lines_l[i]
        h_line_r = h_lines_r[i]
        
        segments_l = re.split('\s+', line_l)[:-1]
        h_segments_l = re.split('\s+', h_line_l)[:-1]
        segments_r = re.split('\s+', line_r)[:-1]
        h_segments_r = re.split('\s+', h_line_r)[:-1]
#        print(segments_l[0] == h_segments_r[0])                #same num?

        img_l = cv2.imread(os.path.join(root_path_l, segments_l[1]))
        img_r = cv2.imread(os.path.join(root_path_r, segments_r[1]))
        img2eye = cv2.imread(os.path.join(two_eye_path, segments_l[1]))
        
        img_l = img_l.transpose(2, 0, 1)
        img_r = img_r.transpose(2, 0, 1)
        img2eye = img2eye.transpose(2, 0, 1)
        imgs_l[i, :, :, :] = img_l.astype(np.float32)
        imgs_r[i, :, :, :] = img_r.astype(np.float32)
        imgs_2eye[i, :, :, :] = img2eye.astype(np.float32)
        labels[i, :] = np.array(list(map(float, [segments_l[2], segments_l[3], h_segments_l[2], h_segments_l[3],
                                              segments_r[2], segments_r[3], h_segments_r[2], h_segments_r[3]])),dtype="float32")        #gaze and head
#        print(labels[i])


    batchsize = 1
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
            f.create_dataset('datal', data=np.array(imgs_l[start:end]).astype(np.float32), **comp_kwargs)
            f.create_dataset('datar', data=np.array(imgs_r[start:end]).astype(np.float32), **comp_kwargs)
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
#
root = '/disks/disk0/linyuqi/dataset/data_gaze/gaze_detection/gaze_2D/'
#root ='F:/datebase/zhang_xucong_paper/MPIIGaze/MPIIGaze/Data/gaze_2D/'
for person in [root + a + '/' for a in os.listdir(root)]:
    for day in [person + a + '/' for a in os.listdir(person)]:
        print(day)
        creat_h5(day)
#        os.path.abspath(day)
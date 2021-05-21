#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 13:08:45 2021

@author: kutalmisince
"""

from PIL import Image
import scipy.io as spio
import torch
import cv2.cv2 as cv
import numpy as np
from torchvision import models, transforms
from collections import namedtuple
import matplotlib.pyplot as plt
from DeepFeatureMatcher import DeepFeatureMatcher

print('asd')

fm = DeepFeatureMatcher()#.to(device)

img_A = (spio.loadmat(file_name='../adam1_4/imgA.mat')['imgA']+128) / 255.0
img_B = (spio.loadmat(file_name='../adam1_4/imgB.mat')['imgB']+128) / 255.0
img_Bw = (spio.loadmat(file_name='../adam1_4/imgBw.mat')['imgBw']+128) / 255.0 

T = transforms.Compose([transforms.ToTensor()])

act_AM = spio.loadmat(file_name='../adam1_4/activationsA.mat')

act_A = []

act_A.append(T(act_AM['activationsA'][5,1]).unsqueeze(0))
act_A.append(T(act_AM['activationsA'][4,1]).unsqueeze(0))
act_A.append(T(act_AM['activationsA'][3,1]).unsqueeze(0))
act_A.append(T(act_AM['activationsA'][2,1]).unsqueeze(0))
act_A.append(T(act_AM['activationsA'][1,1]).unsqueeze(0))
act_A.append(T(act_AM['activationsA'][0,1]).unsqueeze(0))

act_BM = spio.loadmat(file_name='../adam1_4/activationsB.mat')
act_BwM = spio.loadmat(file_name='../adam1_4/activationsBw.mat')

act_B = []

act_B.append(T(act_BwM['activationsBw'][4,1]).unsqueeze(0))
act_B.append(T(act_BwM['activationsBw'][3,1]).unsqueeze(0))
act_B.append(T(act_BwM['activationsBw'][2,1]).unsqueeze(0))
act_B.append(T(act_BwM['activationsBw'][1,1]).unsqueeze(0))
act_B.append(T(act_BwM['activationsBw'][0,1]).unsqueeze(0))
act_B.append(T(act_BM['activationsB'][0,1]).unsqueeze(0))

points_AM = []
points_AM.append(spio.loadmat(file_name='../adam1_4/pointsA_s1_layer_1_2.mat')['pointsA'].T)
points_AM.append(spio.loadmat(file_name='../adam1_4/pointsA_s1_layer_2_2.mat')['pointsA'].T)
points_AM.append(spio.loadmat(file_name='../adam1_4/pointsA_s1_layer_3_2.mat')['pointsA'].T)
points_AM.append(spio.loadmat(file_name='../adam1_4/pointsA_s1_layer_4_2.mat')['pointsA'].T)
points_AM.append(spio.loadmat(file_name='../adam1_4/pointsA_s1_layer_5_2.mat')['pointsA'].T)
points_AM.append(spio.loadmat(file_name='../adam1_4/pointsA_s0.mat')['pointsA'].T)

points_BM = []
points_BM.append(spio.loadmat(file_name='../adam1_4/pointsBw_s1_layer_1_2.mat')['pointsBw'].T)
points_BM.append(spio.loadmat(file_name='../adam1_4/pointsBw_s1_layer_2_2.mat')['pointsBw'].T)
points_BM.append(spio.loadmat(file_name='../adam1_4/pointsBw_s1_layer_3_2.mat')['pointsBw'].T)
points_BM.append(spio.loadmat(file_name='../adam1_4/pointsBw_s1_layer_4_2.mat')['pointsBw'].T)
points_BM.append(spio.loadmat(file_name='../adam1_4/pointsBw_s1_layer_5_2.mat')['pointsBw'].T)
points_BM.append(spio.loadmat(file_name='../adam1_4/pointsB_s0.mat')['pointsB'].T)

args = (act_A, act_B, img_Bw, points_AM, points_BM)

fm.match(img_A, img_B, args)
'''
# compare matlab and python activations
act_A = fm.model.forward(fm.transform(img_A)[0])

map_A5_3 = act_A[5].squeeze(0).numpy().transpose([1,2,0])
map_A5_2 = act_A[4].squeeze(0).numpy().transpose([1,2,0])
map_A4_2 = act_A[3].squeeze(0).numpy().transpose([1,2,0])
map_A3_2 = act_A[2].squeeze(0).numpy().transpose([1,2,0])
map_A2_2 = act_A[1].squeeze(0).numpy().transpose([1,2,0])
map_A1_2 = act_A[0].squeeze(0).numpy().transpose([1,2,0])

# vgg test: load matlab activations and compare results with torch activations, FAIL
act_AM = spio.loadmat(file_name='../adam1_4/activationsA.mat')

act_AM5_3 = act_AM['activationsA'][0,1]
act_AM5_2 = act_AM['activationsA'][1,1]
act_AM4_2 = act_AM['activationsA'][2,1]
act_AM3_2 = act_AM['activationsA'][3,1]
act_AM2_2 = act_AM['activationsA'][4,1]
act_AM1_2 = act_AM['activationsA'][5,1]

print('asd')
'''
'''

act_A = spio.loadmat(file_name='../adam1_4/activationsA.mat')
act_B = spio.loadmat(file_name='../adam1_4/activationsB.mat')
act_Bw = spio.loadmat(file_name='../adam1_4/activationsBw.mat')

T = transforms.Compose([transforms.ToTensor()])

map_AM = T(act_A['activationsA'][0,1]).unsqueeze(0)
map_BM = T(act_B['activationsB'][0,1]).unsqueeze(0)

img_A = spio.loadmat(file_name='../adam1_4/imgA.mat')['imgA']

points_A, points_B = dense_feature_matching(map_AM, map_BM, 1)

fm.plot_keypoints(img_A, (points_A+0.5)*16-0.5)
fm.plot_keypoints(img_B, (points_B+0.5)*16-0.5)

points_AM = spio.loadmat(file_name='../adam1_4_points/dense_pointsA.mat')
points_BM = spio.loadmat(file_name='../adam1_4_points/dense_pointsB.mat')

x = points_AM['pointsA'].transpose()
y = points_BM['pointsB'].transpose()
fm.plot_keypoints(img_A, (x-0.5)*16-0.5)
fm.plot_keypoints(img_B, (y-0.5)*16-0.5)
'''
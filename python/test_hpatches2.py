#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 13:08:45 2021

@author: kutalmisince
"""
from PIL import Image
import numpy as np
import time
from DeepFeatureMatcher import DeepFeatureMatcher

t0 = time.time()

img_A = np.array(Image.open('../data/v_bark_1.ppm'))
img_B = np.array(Image.open('../data/v_bark_6.ppm'))

t1 = time.time()

fm = DeepFeatureMatcher(model = 'VGG19', ratio_th = [0.6, 0.6, 0.8, 0.9, 0.95, 1.0], bidirectional=False)

t2 = time.time()

H, H_init, points_A, points_B = fm.match(img_A, img_B,2)
#H, H_init, pairs_terminal, pairs_adaptive, pairs_final = fm.match(img_A, img_B,2)

t3 = time.time()

elapsed_time = {'image read': t1 - t0, 'construct model': t2 - t1, 'match': t3 - t2}

print(elapsed_time) 
        
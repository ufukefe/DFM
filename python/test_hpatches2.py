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

img_A = np.array(Image.open('../data/1.ppm'))
img_B = np.array(Image.open('../data/4.ppm'))

t1 = time.time()

fm = DeepFeatureMatcher()

t2 = time.time()

H, H_init, points_A, points_B = fm.match(img_A, img_B)

t3 = time.time()

elapsed_time = {'image read': t1 - t0, 'construct model': t2 - t1, 'match': t3 - t2}

print(elapsed_time) 
        
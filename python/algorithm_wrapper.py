#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 14:15:00 2021

@author: kutalmisince
"""
import sys
from PIL import Image 

def algorithm_wrapper(inp_dir, out_dir, pair_list):
    
    print('algorithm :')
    
    for pair in pair_list:
        
        A = Image('inp_dir/'+pair[0]+'.ppm')
        B = Image('inp_dir/'+pair[0]+'.ppm')
        
        points_A, points_B = dfm(A, B)
        
        # save points A and B as a list
    return 

algorithm_wrapper(sys.argv)

#https://github.com/zsdonghao/u-net-brain-tumor/blob/master/train.py


import tensorflow as tf
import tensorlayer as tl
import numpy as np
import os, time

def distort_imgs(data):
    """ data augumentation """
    x1, x2, x3, x4, y = data
    x1, x2, x3, x4, y = tl.prepro.flip_axis_multi([x1, x2, x3, x4, y],
                            axis=0, is_random=True) # up down
    x1, x2, x3, x4, y = tl.prepro.flip_axis_multi([x1, x2, x3, x4, y],
                            axis=1, is_random=True) # left right
    x1, x2, x3, x4, y = tl.prepro.elastic_transform_multi([x1, x2, x3, x4, y],
                            alpha=720, sigma=24, is_random=True)
    x1, x2, x3, x4, y = tl.prepro.rotation_multi([x1, x2, x3, x4, y], rg=20,
                            is_random=True, fill_mode='constant') # nearest, constant
    x1, x2, x3, x4, y = tl.prepro.shift_multi([x1, x2, x3, x4, y], wrg=0.10,
                            hrg=0.10, is_random=True, fill_mode='constant')
    x1, x2, x3, x4, y = tl.prepro.shear_multi([x1, x2, x3, x4, y], 0.05,
                            is_random=True, fill_mode='constant')
    x1, x2, x3, x4, y = tl.prepro.zoom_multi([x1, x2, x3, x4, y],
                            zoom_range=[0.9, 1.1], is_random=True,
                            fill_mode='constant')
    return x1, x2, x3, x4, y



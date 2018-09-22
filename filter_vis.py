from __future__ import absolute_import
from __future__ import print_function

# -*- coding: utf-8 -*-
import numpy as np
import os
from keras.models import load_model
import cv2

os.environ["CUDA_VISIBLE_DEVICES"]="2"
learning_rate=0.001

continue_from_best=False


model=load_model('bestweights3')

os.mkdir('filters')

filters=model.layers[2].layers[0].get_weights()[0].reshape(64,5,5,1)

for c in range(0,63):
    f=filters[c]
    minimum=np.min(f)
    maximum=np.max(f)
    f01=(f-minimum)/(maximum-minimum)
    f255=f01*255
    rf255=cv2.resize(f255,(30,30),interpolation=cv2.INTER_CUBIC)
    cv2.imwrite('filters/'+str(c)+'.png',rf255)
    c=c+1
    
import os
    
for f in os.listdir('filters'):
    g=cv2.imread('filters/'+f,0)
    cv2.imwrite('greyfilters/'+f,g)
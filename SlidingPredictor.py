# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 21:04:38 2018

@author: B
"""

import cv2
import os
import numpy as np
import sys
sys.path.append('/root/tls/Models/')
np.random.seed(123)
import argparse
import Models 
from keras import optimizers

fold=0

parser = argparse.ArgumentParser()
parser.add_argument("--n_classes", type=int, default = 2 )
parser.add_argument("--input_height", type=int , default = 320  )
parser.add_argument("--input_width", type=int , default = 320 )
parser.add_argument("--model_name", type = str , default = "fcn8")
parser.add_argument("--optimizer_name", type = str , default = "sgd")
args = parser.parse_args()

n_classes = args.n_classes
input_height = args.input_height
input_width = args.input_width

optimizer_name = args.optimizer_name
model_name = args.model_name

modelFns = { 'vgg_segnet':Models.VGGSegnet.VGGSegnet , 'vgg_unet':Models.VGGUnet.VGGUnet , 'vgg_unet2':Models.VGGUnet.VGGUnet2 , 'fcn8':Models.FCN8.FCN8 , 'fcn32':Models.FCN32.FCN32   }
modelFN = modelFns[ model_name ]

m = modelFN( n_classes , input_height=input_height, input_width=input_width   )
sgd = optimizers.SGD(lr=0.001)
#adm=optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=5e-05)

m.compile(loss='categorical_crossentropy',
      optimizer= sgd,
      metrics=['accuracy'])

tests=[list(range(25,30)),
       list(range(0,5)),
       list(range(5,10)),
       list(range(10,15)),
       list(range(15,20)),
       list(range(20,25))]

print('loading pages for fold '+str(fold))
pages=[]
for page in tests[fold]:
    pages.append('data/pages/'+str(page)+'.png')
print('test pages are: ')
print(pages)
print("loading weights for fold "+str(fold))
m.load_weights('bestweights'+str(fold))

print ( m.output_shape)

output_height = m.outputHeight
output_width = m.outputWidth

colors=[(255,255,255),(0,0,0)]

outersize=320
trimsize=110
innersize=outersize-2*trimsize

def getImageArr( img , width , height , imgNorm="divide" , odering='channels_first' ):

    try:
        #img = cv2.imread(path, 1)

        if imgNorm == "sub_and_divide":
            img = np.float32(cv2.resize(img, ( width , height ))) / 127.5 - 1
        elif imgNorm == "sub_mean":
            img = cv2.resize(img, ( width , height ))
            img = img.astype(np.float32)
            img[:,:,0] -= 103.939
            img[:,:,1] -= 116.779
            img[:,:,2] -= 123.68
        elif imgNorm == "divide":
            img = cv2.resize(img, ( width , height ))
            img = img.astype(np.float32)
            img = img/255.0

        if odering == 'channels_first':
            img = np.rollaxis(img, 2, 0)
            return img
    except Exception as e:
        print (path)
        print (e)
        img = np.zeros((  height , width  , 3 ))
        if odering == 'channels_first':
            img = np.rollaxis(img, 2, 0)
        return img

def predict(img):
    X = getImageArr(img , args.input_width  , args.input_height  )
    pr = m.predict( np.array([X]) )[0]
    pr = pr.reshape(( output_height ,  output_width , n_classes ) ).argmax( axis=2 )
    seg_img = np.zeros( ( output_height , output_width , 3  ) )
    for c in range(n_classes):
        seg_img[:,:,0] += ( (pr[:,: ] == c )*( colors[c][0] )).astype('uint8')
        seg_img[:,:,1] += ((pr[:,: ] == c )*( colors[c][1] )).astype('uint8')
        seg_img[:,:,2] += ((pr[:,: ] == c )*( colors[c][2] )).astype('uint8')
    seg_img = cv2.resize(seg_img  , (input_width , input_height ))
    return seg_img

print('create predicts folder if does not exist')
if not (os.path.exists('predicts')):
    os.mkdir('predicts')
    
for path in pages:
    page=cv2.imread(path,1)
    rows,cols,ch=page.shape
    x=rows//innersize
    y=cols//innersize
    
    prows=(x+1)*innersize+2*trimsize
    pcols=(y+1)*innersize+2*trimsize
    ppage=np.zeros([prows,pcols,3])
    ppage[trimsize:rows+trimsize,trimsize:cols+trimsize,:]=page[:,:,:]
    pred=np.zeros([rows,cols,3])
    for i in range(0,prows-outersize,innersize):
        for j in range(0,pcols-outersize,innersize):
            patch=ppage[i:i+outersize,j:j+outersize,:]
            ppatch=predict(patch)
            pred[i:i+innersize,j:j+innersize,:]=ppatch[trimsize:trimsize+innersize,trimsize:trimsize+innersize,:]
    cv2.imwrite('predicts/'+path.split('/')[2],pred)

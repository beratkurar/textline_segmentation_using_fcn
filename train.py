# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 13:14:50 2017

@author: B
"""
import sys
sys.path.append('/root/tls/Models/')
import numpy as np
np.random.seed(1006)
import argparse
import Models , PageLoadBatches
from keras.callbacks import ModelCheckpoint
from keras import optimizers
import cv2
import os
import random

fold=3

parser = argparse.ArgumentParser()
parser.add_argument("--n_classes", type=int, default = 2 )
parser.add_argument("--input_height", type=int , default = 320  )
parser.add_argument("--input_width", type=int , default = 320 )
parser.add_argument("--epochs", type = int, default = 250 )
parser.add_argument("--batch_size", type = int, default = 16 )
parser.add_argument("--model_name", type = str , default = "fcn8")
parser.add_argument("--optimizer_name", type = str , default = "sgd")
parser.add_argument("--load_weights", type = str , default = '')

args = parser.parse_args()
train_batch_size = args.batch_size
val_batch_size = args.batch_size
n_classes = args.n_classes
input_height = args.input_height
input_width = args.input_width
epochs = args.epochs
load_weights = args.load_weights
optimizer_name = args.optimizer_name
model_name = args.model_name


trains=[list(range(0,20)),
        list(range(5,25)),
        list(range(10,30)),
        [15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,0,1,2,3,4],
        [20,21,22,23,24,25,26,27,28,29,0,1,2,3,4,5,6,7,8,9],
        [25,26,27,28,29,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]]
validations=[list(range(20,25)),
             list(range(25,30)),
             list(range(0,5)),
             list(range(5,10)),
             list(range(10,15)),
             list(range(15,20))]
tests=[list(range(25,30)),
       list(range(0,5)),
       list(range(5,10)),
       list(range(10,15)),
       list(range(15,20)),
       list(range(20,25))]

print('training for fold '+str(fold))

if not (os.path.exists('ptrain'+str(fold))):   
    print('create patch folders if does not exist')    
    os.mkdir('ptrain'+str(fold))
    os.mkdir('pltrain'+str(fold))
    os.mkdir('pvalidation'+str(fold))
    os.mkdir('plvalidation'+str(fold))
    print('train and validation patch folders are generated')
    print('train patches are being generated')
    patchSize=320
    pages=[]
    labels=[]
    for page in trains[fold]:
        pages.append('data/pages/'+str(page)+'.png')
    for label in trains[fold]:
        labels.append('data/labels/'+str(label)+'.png')
    print('train pages are: ')
    print(pages)
    print('train labels are: ')
    print(labels)
    i=0
    while (i <50000):
        page_number=random.randint(0,19)
        page_name=pages[page_number]
        label_name=labels[page_number]
        page=cv2.imread(page_name,0)
        lpage=cv2.imread(label_name,0)
        rows,cols=page.shape
        x=random.randint(0,rows-patchSize)
        y=random.randint(0,cols-patchSize)
        patch=page[x:x+patchSize,y:y+patchSize]
        cv2.imwrite('ptrain'+str(fold)+'/'+page_name.split('/')[2][:-4]+"_patch"+str(i)+".png",patch)    
        lpatch=lpage[x:x+patchSize,y:y+patchSize]
        cv2.imwrite('pltrain'+str(fold)+'/'+label_name.split('/')[2][:-4]+"_patch"+str(i)+".png",lpatch)
        i=i+1
    print(str(i)+' train patches for fold '+str(fold)+ ' is generated')
    print('validation patches are being generated')
    pages=[]
    labels=[]
    for page in validations[fold]:
        pages.append('data/pages/'+str(page)+'.png')
    for label in validations[fold]:
        labels.append('data/labels/'+str(label)+'.png')
    print('validation pages are: ')
    print(pages)
    print('validation labels are: ')
    print(labels)
    i=0
    while (i <6000):
        page_number=random.randint(0,4)
        page_name=pages[page_number]
        label_name=labels[page_number]
        page=cv2.imread(page_name,0)
        lpage=cv2.imread(label_name,0)
        rows,cols=page.shape
        x=random.randint(0,rows-patchSize)
        y=random.randint(0,cols-patchSize)
        patch=page[x:x+patchSize,y:y+patchSize]
        cv2.imwrite('pvalidation'+str(fold)+'/'+page_name.split('/')[2][:-4]+"_patch"+str(i)+".png",patch)        
        lpatch=lpage[x:x+patchSize,y:y+patchSize]
        cv2.imwrite('plvalidation'+str(fold)+'/'+label_name.split('/')[2][:-4]+"_patch"+str(i)+".png",lpatch)
        i=i+1
    print(str(i)+' validation patches for fold '+str(fold)+ ' is generated')

modelFns = { 'vgg_segnet':Models.VGGSegnet.VGGSegnet , 'vgg_unet':Models.VGGUnet.VGGUnet , 'vgg_unet2':Models.VGGUnet.VGGUnet2 , 'fcn8':Models.FCN8.FCN8 , 'fcn32':Models.FCN32.FCN32   }
modelFN = modelFns[ model_name ]

m = modelFN( n_classes , input_height=input_height, input_width=input_width   )
sgd = optimizers.SGD(lr=0.001)
#adm=optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=5e-05)

m.compile(loss='categorical_crossentropy',
      optimizer= sgd,
      metrics=['accuracy'])

if len( load_weights ) > 0:
    print("loading initial weights")
    m.load_weights(load_weights)

print ( m.output_shape)

output_height = m.outputHeight
output_width = m.outputWidth

G  = PageLoadBatches.imageSegmentationGenerator( 'ptrain'+str(fold)+'/', 'pltrain'+str(fold)+'/' ,  train_batch_size,  n_classes , input_height , input_width , output_height , output_width   )

G2  = PageLoadBatches.imageSegmentationGenerator( 'pvalidation'+str(fold)+'/' , 'plvalidation'+str(fold)+'/' ,  val_batch_size,  n_classes , input_height , input_width , output_height , output_width   )

mcp=ModelCheckpoint( filepath='bestweights'+str(fold), monitor='val_loss', save_best_only=True, save_weights_only=True,verbose=1)

for ep in range( epochs ):
    print ('epoch:'+str(ep))
    m.fit_generator( G , 3125, validation_data=G2 , validation_steps=375,  epochs=1,callbacks=[mcp] )


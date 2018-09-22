# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 19:39:03 2018

@author: B
"""

import cv2
import os

for page in os.listdir('orgpages/'):
    img=cv2.imread('orgpages/'+page,0)
    inv=255-img
    cv2.imwrite('pages/'+page.split('.')[0]+'.png',inv)
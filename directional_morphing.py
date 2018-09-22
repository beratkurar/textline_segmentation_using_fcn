import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

def extract_contours_info(img, min_area=100, vis=None):
    im2, contours, hierarchy1 = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    res_countours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < min_area or len(cnt) < 15:
            continue

        [vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
        hull = cv2.convexHull(cnt)

        center, (MA, ma), angle = cv2.fitEllipse(hull)

        res_countours.append((cnt, np.asarray((vx[0], vy[0])), ma / (ma + MA)))

        if vis is not None:
            cv2.line(vis, (int(x), int(y)), (int(x + 100 * vx), int(y + 100 * vy)), (0, 0, 200), 10)
            cv2.drawContours(vis, [hull], 0, (0, 255, 0), 5)

    return res_countours


def segment(img, contours_info, v, epsilon=0.2):
    mask = np.zeros((img.shape[0], img.shape[1]))

    n = len(contours_info)
    for i in range(n):
        ori = contours_info[i][1]

        val = ((contours_info[i][2])**2) * (np.abs(np.inner(v, ori)))

        if val < epsilon:
            cv2.fillPoly(mask, [contours_info[i][0]], 1)

    return mask


def connect_segmented(img, N=50, kernel_size=(50, 50), epsilon=0.2, vis=None):
    contours = extract_contours_info(img, vis=vis)

    step = (np.pi) / N

    final_img = np.zeros(img.shape, dtype=np.bool)

    for k in range(N):
        print('step : ', k)
        theta = k * step
        c, s = np.cos(theta), np.sin(theta)
        v = np.asarray((c, s))

        mask = segment(img, contours, v, epsilon=epsilon)


        kernel = np.zeros(kernel_size, np.uint8)

        c, s = np.cos(np.pi/2 + theta), np.sin(np.pi/2 + theta)
        pv = np.asarray((c, s))

        cv2.line(kernel, (int(kernel.shape[0] / 2 * (1 - pv[0])), int(kernel.shape[1] / 2 * (1 - pv[1]))),
                 (int(kernel.shape[0] / 2 * (1 + pv[0])), int(kernel.shape[1] / 2 * (1 + pv[1]))), 1, 1)

        mask = cv2.dilate(mask, kernel, iterations=1)

        final_img = np.bitwise_or(final_img, np.asarray(mask, dtype=np.bool))

        #cv2.imshow('mask', mask)
        #cv2.imshow('kernel', 255*kernel)
        #cv2.waitKey(1)

    final_img = np.asarray(final_img, dtype=np.uint8)*255

    return final_img



for img_path in os.listdir('predicts'):
    img = cv2.imread('predicts/'+img_path, 0)
    refined_img = connect_segmented(img, N=10, kernel_size=(20, 20), epsilon=0.2)

    cv2.imwrite('./post/{}'.format(img_path), refined_img)

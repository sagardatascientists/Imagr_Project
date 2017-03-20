from compiler.misc import flatten

from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import cv2
from scipy.misc import imread
import os

import numpy as np

def applyImagePreprocessing(imag_path):
    img_roi = []
    im = cv2.imread(imag_path)
    image = cv2.resize(im, (600, 600), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edged = cv2.Canny(blurred, 50, 200, 255)
    image, contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if (w >= 100 and w <= 250) and (h <= 120 and h >= 10):
            # print w, h
            #  print cv2.contourArea(c)
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 2)
            roi = image[y:y + h, x:x + w]
            resized_roi = cv2.resize(roi, (64, 32))
            resized_roi = resized_roi.reshape((1, 2048))
            resized_roi = np.float32(resized_roi)
            if len(img_roi)==10: break
            img_roi.append(resized_roi)
            # cv2.imwrite(op+label+"_"+str(idx)+".jpg", resized)
            # cv2.putText(image,'Find',(x+w+10,y+h),0,0.3,(0,0,255))
            # cv2.imshow("Show",image)
            # idx=idx+1
            # cv2.waitKey()

    size = len(img_roi)
    if len(img_roi) !=0:
        while size<10:
            img_roi.append(img_roi[0])
            size=size+1
    return img_roi

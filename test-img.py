import os
import cv2 as cv
import numpy as np

PATH = "./img/"

IMAGES = os.listdir(PATH)

def get_image():
    return cv.imread(PATH+IMAGES[3])

def refresh(x):
    img = get_image()
    t1 = cv.getTrackbarPos("t1", "params")
    a = cv.getTrackbarPos("a", "params")

    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    img = cv.Canny(img, t1, t1)
    
    c, h = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    imgs = []
    for cnt in c:
        area = cv.contourArea(cnt)

        if area > a:
            cv.drawContours(img, cnt, -1, (255, 0, 255), 5)
            peri = cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, 0.02*peri, True)
            x,y,w,h = cv.boundingRect(approx)
            imgs.append(get_image()[x:w+x, y:y+h])

    cv.imshow("imgs", img)


cv.namedWindow("params")
cv.resizeWindow("params", 600, 300)
cv.createTrackbar("t1", "params", 255, 500, refresh)
cv.createTrackbar("a", "params", 1000, 10000, refresh)

original_img = get_image()

cv.waitKey(0)
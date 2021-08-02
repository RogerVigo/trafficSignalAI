import cv2
import pandas as pd
import numpy as np
from time import time, time_ns
import os
import cv2 as cv
import numpy as np
import pickle as pkl
from PIL import Image
import tensorflow as tf

def show_image(image):
    cv.imshow("image", image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def run():
    video = cv.VideoCapture(0)
    ret, frame = video.read()
    shape = frame.shape
    print(shape)
    kernel = 166
    x = 0
    y = 0
    while True:
        ret, image = video.read()
        if x+kernel <= shape[0]:
            while y+kernel <= shape[1]:
                crop = image[x:x+kernel, y:y+kernel, :]
                #self.show_image(crop*255)
                final = np.array([np.resize(crop, (32,32,3))])
                #result = self.model.predict(final)
                #classes = self.model.predict_classes(final)
                #if np.amax(result) >= self.threshold:
                #    original_image = self.draw_matches(original_image, (x,y), kernel)
                    #print(self.labels[classes[0]])
                y += kernel//2

            x += kernel//2
            y = 0
        else:
            x = 0
            y = 0
            show_image(image)


def load_images(path):
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []
    valid_images = []
    valid_labels = []
    # r=root, d=directories, f = files
    target = np.sort(np.array(os.listdir(path), dtype="int16"))

    for i in target:
        images_path = path + "/" + str(i) + "/"
        dir_len = len(os.listdir(images_path))

        for x, img in enumerate(os.listdir(images_path)):
            loaded = cv.imread(images_path + img)

            if x <= dir_len * 0.8:
                train_images.append(loaded)
            else:
                test_images.append(loaded)

    return train_images, test_images, target

path = "./data/myData/"

tr, ts, tg = load_images(path)
x = np.array(tr)
show_image(x[0])
y = tf.random.shuffle(x, tf.random.set_seed(1234))
y = np.array(y)
show_image(y[0])
#print(len(images))
#run()
cv.destroyAllWindows()
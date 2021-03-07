from data import datasets_preparing as dataset
import pickle
import cv2 as cv


label_names_file = "data/label_names.csv"

for i in range(9):
    with open("data/data" + str(i) + ".pickle", "rb") as f:
        d = pickle.load(f, encoding='latin1')  # dictionary type, we use 'latin1' for python3

        for k, v in d:
            print(k)

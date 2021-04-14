import pandas as pd
import numpy as np
from time import time, time_ns
import os
import cv2 as cv

def show_image(image):
    cv.imshow("image", image)
    cv.waitKey(50)
    #cv.destroyAllWindows()

x =[
    [
        [[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3]],
        [[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3]],
        [[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3]]
    ],
    [
        [[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3]],
        [[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3]],
        [[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3]]
    ]
]
print(np.shape(x))
exit()

image = cv.imread("/home/roger/Imágenes/signals/carretera.png", cv.IMREAD_COLOR)
shape = np.shape(image)
final_image = []
kernel = 100
x = 0
y = 0
print(shape)
start = time()
while True:
    b = False
    for i in range(x, x + kernel):
        if y + kernel < shape[1]:
            final_image.append(image[i][y:kernel+y])
        else:
            b = True
    if b:
        print("a")
        x = x + kernel//2
        y = 0
        if x + kernel > shape[0]:
            print(x+kernel)
            print(shape)
            break

        continue
    final_image = np.array(final_image)
    #show_image(final_image)
    final_image = []
    y = y + kernel//2

end = time()

print(f"Total Time: {(end-start)}")
cv.destroyAllWindows()
exit()


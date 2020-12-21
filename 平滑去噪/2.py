import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
import random
import cv2
import scipy.misc
import scipy.signal
import scipy.ndimage

def saltpepper_noise(image, prob):
    output = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

def gussian_noise(image, mean=0, var=0.001):
    image = np.array(image / 255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out * 255)
    return out

def pulse_noise(image, prob):
    out2 = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                out2[i][j] = 255
            elif rdn > thres:
                out2[i][j] = 255
            else:
                out2[i][j] = image[i][j]
    return out2
image = np.array(Image.open('C:/Users/fengyuting/Desktop/lena.jpg'))
###加噪处理，分别加椒盐 高斯 脉冲噪声
output = saltpepper_noise(image, 0.002)#椒盐噪声
out = gussian_noise(image, mean=0, var=0.001)#高斯噪声
out2 = pulse_noise(image, 0.002)#脉冲噪声
###show 加噪后的图片
cv2.imshow("gussian", out)
cv2.imshow("saltpepper", output)
cv2.imshow("pulse", out2)
cv2.waitKey(0)

###不同boxfilter 对三种加噪后的line进行处理
kernelsizes = [(3, 3), (9, 9), (15, 15)]
for kernel in kernelsizes:
    blur = cv2.boxFilter(out, -1, kernel, anchor=(-1, -1), normalize=True, borderType=cv2.BORDER_DEFAULT)
    cv2.imshow('boxfilter gussian : ' + str(kernel), blur)
cv2.waitKey(0)
for kernel in kernelsizes:
    blur = cv2.boxFilter(out2, -1, kernel, anchor=(-1, -1), normalize=True, borderType=cv2.BORDER_DEFAULT)
    cv2.imshow('boxfilter pluse : ' + str(kernel), blur)
cv2.waitKey(0)
for kernel in kernelsizes:
    blur = cv2.boxFilter(output, -1, kernel, anchor=(-1, -1), normalize=True, borderType=cv2.BORDER_DEFAULT)
    cv2.imshow('boxfilter saltpepper : ' + str(kernel), blur)
cv2.waitKey(0)
###不同gasuss filter对三种加噪后的图片处理
gaussian = cv2.GaussianBlur(out, (3, 3), 1)
cv2.imshow('gussianfilter gussian :(3, 3)', gaussian)
gaussian = cv2.GaussianBlur(out, (3, 3), 0)
cv2.imshow('gussianfilter gussian sigma0 : (3, 3)' , gaussian)
cv2.waitKey(0)
gaussian = cv2.GaussianBlur(out2, (3, 3), 1)
cv2.imshow('gussianfilter pluse : (3, 3)' , gaussian)
gaussian = cv2.GaussianBlur(out, (3, 3), 0)
cv2.imshow('gussianfilter pluse sigma0 : (3, 3)' , gaussian)
cv2.waitKey(0)
gaussian = cv2.GaussianBlur(output, (3, 3), 1) 
cv2.imshow('gussianfilter saltpepper : (3, 3)', gaussian)
gaussian = cv2.GaussianBlur(out, (3, 3), 0)
cv2.imshow('gussianfilter saltpepper sigma0 : (3, 3)' , gaussian)
cv2.waitKey(0)

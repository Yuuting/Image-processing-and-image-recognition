import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
import random
import cv2
import scipy.misc
import scipy.signal
import scipy.ndimage

def kernal(img,kernal):
    h,w = img.shape
    edge_img=np.zeros([h-2,w-2])
    for i in range(h-2):
        for j in range(w-2):
            edge_img[i,j]=img[i,j]*kernal[0,0]+img[i,j+1]*kernal[0,1]+img[i,j+2]*kernal[0,2]+img[i+1,j]*kernal[1,0]+img[i+1,j+1]*kernal[1,1]+img[i+1,j+2]*kernal[1,2]+img[i+2,j]*kernal[2,0]+img[i+2,j+1]*kernal[2,1]+img[i+2,j+2]*kernal[2,2]
    return edge_img

def prewitt(img):
    gray = cv2.imread(img,0)
    h = gray.shape[0]
    w = gray.shape[1]
    x_prewitt=np.array([[1,0,-1],
                       [1,0,-1],
                       [1,0,-1]])
    y_prewitt=np.array([[1,1,1],
                       [0,0,0],
                       [-1,-1,-1]])
 
    img=np.zeros([h+2,w+2])
    img[2:h+2,2:w+2]=gray[0:h,0:w]
    edge_x_img=kernal(img,x_prewitt)
    edge_y_img=kernal(img,y_prewitt) 
 
    edge_img_sqrt=np.zeros([h,w],np.uint8)
    for i in range(h):
        for j in range(w):
            edge_img_sqrt[i][j]=np.sqrt((edge_x_img[i][j])**2+(edge_y_img[i][j])**2)

    return edge_img_sqrt

def sobel(img):
    #定义不同方向的梯度算子
    x_sobel = np.array([[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]])
    y_sobel = np.array([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]])
    gray_img = cv2.imread(img,0)
    h, w = gray_img.shape
    img = np.zeros([h + 2, w + 2])
    img[2:h + 2, 2:w + 2] = gray_img[0:h, 0:w]
    x_edge_img = kernal(img, x_sobel)
    y_edge_img = kernal(img, y_sobel)
    edge_img = np.zeros([h, w],np.uint8)
    for i in range(h):
        for j in range(w):
            edge_img[i][j] = np.sqrt(x_edge_img[i][j] ** 2 + y_edge_img[i][j] ** 2) 
    return edge_img

def log(img):
 
    #第一步灰度化
    gray_img = cv2.imread(img,0)
 
    #第二步高斯滤波
    #高斯算子
    g_filter=np.array([[0,0,1,0,0],
                       [0,1,2,1,0],
                       [1,2,16,2,1],
                       [0,1,2,1,0],
                       [0,0,1,0,0]])
    self_g_img=np.pad(gray_img,((2,2),(2,2)),'constant')#扩展操作
    #以下进行的其实就是滤波操作
    w,h=self_g_img.shape
    for i in range(w-4):
        for j in range(h-4):
            self_g_img[i][j]=np.sum(self_g_img[i:i+5,j:j+5]*g_filter)
 
    # 第三步：计算laplace二阶导数，操作和laplace算子一样
    lap4_filter = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])  # 4邻域laplacian算子
    g_pad=np.pad(self_g_img,((1,1),(1,1)),'constant')
    # 4邻域
    edge4_img = np.zeros((w, h))
    for i in range(w - 2):
        for j in range(h - 2):
            edge4_img[i, j] = np.sum(g_pad[i:i + 3, j:j + 3] * lap4_filter)
            if edge4_img[i, j] < 0:
                edge4_img[i, j] = 0  # 把所有负值修剪为0
 
    lap8_filter = np.array([[0, 1, 0], [1, -8, 1], [0, 1, 0]])  # 8邻域laplacian算子
    # 8邻域
    g_pad = np.pad(self_g_img, ((1, 1), (1, 1)), 'constant')
    edge8_img = np.zeros((w, h))
    for i in range(1,w - 1):
        for j in range(1,h - 1):
            edge8_img[i, j] = np.sum(g_pad[i-1:i + 2, j-1:j + 2] * lap8_filter)
            if edge8_img[i, j] < 0:
                edge8_img[i, j] = 0
    return [edge4_img,edge8_img]

def canny(img):
    img = cv2.imread(img, 0)
    img = cv2.GaussianBlur(img, (3, 3), 2)
    edge_img = cv2.Canny(img, 50,100)
    return edge_img

img = 'C:/Users/fengyuting/Desktop/lena.jpg'
edge1 = prewitt(img)
cv2.imshow('prewitt',edge1)
cv2.waitKey(0)

edge2 = sobel(img)
cv2.imshow('sobel',edge2)
cv2.waitKey(0)

[edge3,edge4] = log(img)
cv2.imshow('lap4_filter',edge3)
cv2.waitKey(0)
cv2.imshow('lap8_filter',edge4)
cv2.waitKey(0)

edge5 = canny(img)
cv2.imshow('canny',edge5)
cv2.waitKey(0)

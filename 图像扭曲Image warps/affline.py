import cv2
import numpy as np

img = cv2.imread('lena.jpg')
rows, cols, depth = img.shape

pts1 = np.float32([[50, 50],[200, 50],[50,200]])
pts2 = np.float32([[10,100],[200,50],[100,250]])

M = cv2.getAffineTransform(pts1, pts2)

forward_warp_pic = np.zeros_like(img)
inverse_warp_pic = np.zeros_like(img)

for d in range(depth):
    for v in range(rows):
        for u in range(cols):
            x = int(round(M[0,0]*u + M[0,1]*v + M[0,2]))
            y = int(round(M[1,0]*u + M[1,1]*v + M[1,2]))
            if x < 0 or x >= cols or y < 0 or y >= rows:
                continue
            forward_warp_pic[y,x,d] = img[v,u,d]

inverse_warp_pic = cv2.warpAffine(img, M, (rows, cols))

cv2.imshow('original', img)
cv2.waitKey(0)

cv2.imshow('forward_warp_pic', forward_warp_pic)
cv2.waitKey(0)

cv2.imshow('inverse_warp_pic', inverse_warp_pic)
cv2.waitKey(0)
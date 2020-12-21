import numpy as np
import cv2 as cv
import math

def demosaic(img):
    img_demosaic = cv.cvtColor(img, cv.COLOR_BayerRG2BGR)
    return img_demosaic

def WBC(img):
    # 定义BGR通道
    r = img[:, :, 2]
    b = img[:, :, 0]
    g = img[:, :, 1]

    # 计算BGR均值
    averB = np.mean(b)
    averG = np.mean(g)
    averR = np.mean(r)

    # 计算灰度均值
    grayValue = (averR + averB + averG) / 3

    # 计算增益
    kb = grayValue / averB
    kg = grayValue / averG
    kr = grayValue / averR

    # 补偿通道增益
    r[:] = r[:] * kr
    g[:] = g[:] * kg
    b[:] = b[:] * kb
    return img

def BGRtoRGB(img):
    RGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return RGB

def gamma_trans(img, gamma):  # gamma函数处理
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]  # 建立映射表
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)  # 颜色值为整数
    return cv.LUT(img, gamma_table)  # 图片颜色查表。另外可以根据光强（颜色）均匀化原则设计自适应算法。


def Denoising(img):
    dst = cv.fastNlMeansDenoising(img, None, 10, 7, 21)
    return dst

#文件路径在桌面
img = cv.imread("C:/Users/fengyuting/Desktop/raw-data-BayerpatternEncodedImage.tif",0) 
cv.namedWindow("Image",cv.WINDOW_NORMAL)
cv.imshow("Image",img)
cv.waitKey(0)
cv.destroyAllWindows() 

#去马赛克
img_demosaic = demosaic(img)
cv.namedWindow("img_demosaic",cv.WINDOW_NORMAL)
cv.imshow("img_demosaic", img_demosaic)
cv.waitKey(0)
cv.destroyAllWindows()

#白平衡
WBC_img = WBC(img_demosaic)
cv.namedWindow("WBC",cv.WINDOW_NORMAL)
cv.imshow("WBC", WBC_img)
cv.waitKey(0)
cv.destroyAllWindows()

#颜色空间
RGB_img = BGRtoRGB(WBC_img)
cv.namedWindow("Color_correction",cv.WINDOW_NORMAL)
cv.imshow("Color_correction", RGB_img)
cv.waitKey(0)
cv.destroyAllWindows()

#伽马校正
mean = np.mean(WBC_img)
gamma_val = math.log10(0.5) / math.log10(mean / 255)  # 公式计算gamma
image_gamma_correct = gamma_trans(RGB_img, gamma_val)  # gamma变换
cv.namedWindow("image_gamma_correct",cv.WINDOW_NORMAL)
cv.imshow("image_gamma_correct", image_gamma_correct)
cv.waitKey(0)
cv.destroyAllWindows()

#去噪
Denoising_img = Denoising(image_gamma_correct)
cv.namedWindow('Denoising', cv.WINDOW_NORMAL)
cv.imshow("Denoising", Denoising_img)
cv.waitKey(0)
cv.destroyAllWindows()
'''

#去马赛克
img_demosaic = demosaic(img)
cv.namedWindow("img_demosaic",cv.WINDOW_NORMAL)
cv.imshow("img_demosaic", img_demosaic)
cv.waitKey(0)
cv.destroyAllWindows()

#去噪
Denoising_img = Denoising(img_demosaic)
cv.namedWindow('Denoising', cv.WINDOW_NORMAL)
cv.imshow("Denoising", Denoising_img)
cv.waitKey(0)
cv.destroyAllWindows()

#白平衡
WBC_img = WBC(Denoising_img)
cv.namedWindow("WBC",cv.WINDOW_NORMAL)
cv.imshow("WBC", WBC_img)
cv.waitKey(0)
cv.destroyAllWindows()

#颜色空间
RGB_img = BGRtoRGB(WBC_img)
cv.namedWindow("Color_correction",cv.WINDOW_NORMAL)
cv.imshow("Color_correction", RGB_img)
cv.waitKey(0)
cv.destroyAllWindows()

#伽马校正
mean = np.mean(WBC_img)
gamma_val = math.log10(0.5) / math.log10(mean / 255)  # 公式计算gamma
image_gamma_correct = gamma_trans(RGB_img, gamma_val)  # gamma变换
cv.namedWindow("image_gamma_correct",cv.WINDOW_NORMAL)
cv.imshow("image_gamma_correct", image_gamma_correct)
cv.waitKey(0)
cv.destroyAllWindows()
'''

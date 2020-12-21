# Image-processing-and-image-recognition
2020秋北邮研究生图像处理课程代码总结

## 相机数字图像处理

在网上任意找一个相机里.raw的图片，使用课程中所提到的数字图像处理技术，进行任意两种组合的操作，

（举例：组合一 去马赛克->白平衡->color correction->gamma correction->去噪

   组合二 去马赛克->去噪->白平衡->color correction->gamma correction，鼓励使用课堂上未详细讲解的操作）

观察得到最终JPEG图片的差异，并论述差异。

1）上交代码 （python，作业中需要明确标明main.py，以便Python main.py即可运行结果）

2）附回答文档，总结具体操作，及不同组合的操作对结果的影响。

## 平滑去噪 (python coding)

在lena.jpg图上做如下操作：

1.得出三张噪声图：添加高斯噪声、椒盐噪声、脉冲噪声；

2.用不同尺寸的box filter对三张噪声图进行去噪操作；

3.用不同sigma的高斯噪声对三张噪声图进行去噪操作。

## 图像边缘检测

对lena.jpg做以下几种边缘检测： 

\- 基于一阶梯度求导：Prewitt和Sobel两种算子 

\- 基于LoG算法 

\- Canny检测算法-可选操作如下：1.Single-threhold 2. double threshold. 可以使用Python自带的各种库函数

## 图像融合 Image blending

Step1 Generate Laplacian pyramid Lo of orange image. Generate the Laplacian pyramid La of apple image. 

Step 2&3 Generate Laplacian pyramid Lc by 

– copying left half of the nodes at each level from apple and 

– right half of nodes from orange pyramids. 

Step 4 Reconstruct a combined image from Lc.

## 图像检索Image retrieval

实现图像检索的过程。

1.特征提取（Feature extraction）：使用SIFT或者HoG

2.特征融合(Feature aggregation)可跳过此步，也可将SIFT或者HoG特征的拼接

3.图像匹配（Image matching）：计算查询图像特征与数据集(images.zip)所有图片特征的欧式距离，且将距离排序。

## 计算视频序列的光流

根据opencv自带的Lucas-Kanade函数： cv.CalcOpticalFlowLK 或者 cv.calcOpticalFlowPyrLK，计算“shibuya.mp4”视频序列的光流

## 图像扭曲Image warps

1.证明题：仿射变换(Affine Transformation)中平行线变换后仍然是平行线 

2.编程题：通过实验对比正向变换(Forward warping)与反向变换（inverse warping）对图像变形/扭曲(Image warps)结果的不同，且总结正向变换的缺点可能有哪些。

注：  pts1 = np.float32([[50,50],[200,50],[50,200]]) 

​     pts2 = np.float32([[10,100],[200,50],[100,250]]) 

​     以pts1->pts2的变换矩阵为对lena.jpg扭曲所需的变换关系

## 目标跟踪

在不使用mean-shift 库函数的情形下，采用mean-shift算法实现目标跟踪，给定第一帧ms_1.jpg，跟踪该图中“任意一个区域”在第二帧image/ms_2.jpg中的位置。

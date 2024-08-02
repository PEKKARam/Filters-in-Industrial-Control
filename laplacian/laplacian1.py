import cv2

import numpy as np

# 图像锐化处理函数

def laplacian_sharpening(img, K_size=3):

    # 图像高H，宽W

    H, W = img.shape

    # 零填充，在图像边缘添加额外的像素，以便滤波器可以处理图像边缘

    pad = K_size // 2

    out = np.zeros((H + pad * 2, W + pad * 2), dtype=float)

    out[pad: pad + H, pad: pad + W] = img.copy().astype(float)

    # 将处理后的图像像素值复制到填充图像
    tmp = out.copy()

    # 拉普拉斯滤波器卷积核表示

    K = [[0., 1., 0.],[1., -4., 1.], [0., 1., 0.]]
    # K = [[1., 1., 1.],[1., -8., 1.], [1., 1., 1.]]


    # 滤波和锐化图像

    for y in range(H):

        for x in range(W):

            # 提取当前像素周围的子区域，卷积，求和，锐化，将滤波响应加回原始图像的像素值，结果储存到out相关位置

            out[pad + y, pad + x] = (-1) * np.sum(K * (tmp[y: y + K_size, x: x + K_size])) + tmp[pad + y, pad + x]

    out = np.clip(out, 0, 255)

    out = out[pad: pad + H, pad: pad + W].astype(np.uint8)

    return out

# 读取灰度图像
img = cv2.imread("./laplacian/test.jpg",0).astype(float)

# 通过拉普拉斯滤波器进行图像锐化

out = laplacian_sharpening(img, K_size=3)

# 保存结果

cv2.imwrite("1out.jpg", out)

cv2.imshow("result", out)

cv2.waitKey(0)

cv2.destroyAllWindows()
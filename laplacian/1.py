import cv2
import numpy as np
from scipy.ndimage import convolve
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
# scikit-image
# cv2
# numpy
# scipy

# 图像锐化
def laplacian_sharpening(img, kernel):
    # 使用卷积函数进行图像处理
    out = convolve(img, kernel, mode='reflect')

    # 将滤波响应加回原始图像的像素值
    out = img - out

    # 将结果限制在 [0, 255] 范围内
    out = np.clip(out, 0, 255)

    return out.astype(np.uint8)

#计算图像梯度熵 
def gradient_entropy(img):
    gx = np.gradient(img, axis=0)
    gy = np.gradient(img, axis=1)
    grad_magnitude = np.sqrt(gx**2 + gy**2)
    hist, _ = np.histogram(grad_magnitude, bins=256)
    hist = hist / np.sum(hist)
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    return entropy

# 读取灰度图像
img = cv2.imread("./laplacian/Test1.jpg", 0).astype(float)

# 定义拉普拉斯滤波器卷积核
K1 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
K2 = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])  
K3 = np.array([[0, 0, -1, 0, 0], 
               [0, -1, -2, -1, 0], 
               [-1, -2, 16, -2, -1], 
               [0, -1, -2, -1, 0], 
               [0, 0, -1, 0, 0]])

# 生成锐化后的图像
out1 = laplacian_sharpening(img, K1)
out2 = laplacian_sharpening(img, K2)
out3 = laplacian_sharpening(img, K3)

# 评估函数
# PSNR峰值信噪比
# SSIM结构相似性
# entropy梯度熵
def valuate(img,out):
    # 将图像转换为相同的数据类型
    img = img.astype(np.float32)
    out = out.astype(np.float32)
    data_range = out.max() - out.min()

    psnr_value = psnr(img, out, data_range=data_range)
    ssim_value = ssim(img, out, data_range=data_range)
    entropy_value = gradient_entropy(out)
    print(f"PSNR={psnr_value}, SSIM={ssim_value}, Entropy={entropy_value}")


# 打印评估结果
print(f"K1")
valuate(img,out1)
print(f"K2")
valuate(img,out2)
print(f"K3")
valuate(img,out3)

# 将三张图像垂直拼接在一起
combined = np.hstack((out1, out2, out3))

# 保存结果
cv2.imwrite("./laplacian/combined.jpg", combined)
cv2.imshow("result", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()

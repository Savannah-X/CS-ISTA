import numpy as np
import cv2
import matplotlib.pyplot as plt
from phantominator import shepp_logan
import pywt
from skimage.metrics import structural_similarity as ssim

# 读取CT图像
ct_image = cv2.imread(r"hbh.BMP", cv2.IMREAD_GRAYSCALE)
if ct_image is None:
    raise FileNotFoundError("Image not found at specified path.")
ct_image = cv2.resize(ct_image, (512, 512))

# 生成Shepp-Logan phantom
phantom = shepp_logan(512)

# 计算2D-FFT并获取幅度谱
def compute_k_space(image):
    k_space = np.fft.fft2(image)
    k_space_magnitude = np.log(np.abs(k_space) + 1)
    return k_space, k_space_magnitude

k_space_ct, k_space_magnitude_ct = compute_k_space(ct_image)
k_space_phantom, k_space_magnitude_phantom = compute_k_space(phantom)

# 稀疏采样函数
def sparse_sampling(k_space, sampling_rate):
    total_points = k_space.size
    num_samples = int(total_points * sampling_rate)
    
    indices = np.random.choice(total_points, num_samples, replace=False)
    sparse_k_space = np.zeros_like(k_space)
    sparse_k_space.flat[indices] = k_space.flat[indices]
    
    return sparse_k_space

# 设计稀疏采样模式
sampling_rates = [0.25, 0.5, 0.75]
sparse_k_spaces_ct = {rate: sparse_sampling(k_space_ct, rate) for rate in sampling_rates}
sparse_k_spaces_phantom = {rate: sparse_sampling(k_space_phantom, rate) for rate in sampling_rates}

# PSNR计算
def psnr(original, reconstructed):
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

# ISTA算法
def initialize_reconstruction(sparse_k_space):
    if np.all(sparse_k_space == 0):
        raise ValueError("Input sparse k-space is empty.")
    
    return np.fft.ifft2(sparse_k_space).real

def ista_reconstruction(sparse_k_space, max_iter=1000, lambda_reg=0.1, tol=1e-5):
    reconstructed_image = initialize_reconstruction(sparse_k_space)
    previous_image = reconstructed_image.copy()
    
    for i in range(max_iter):
        coeffs = pywt.wavedec2(reconstructed_image, 'haar', level=3)
        
        # 软阈值处理
        coeffs_thresholded = list(coeffs)
        coeffs_thresholded[0] = pywt.threshold(coeffs[0], lambda_reg, mode='soft')
        coeffs_thresholded[1] = tuple(pywt.threshold(c, lambda_reg, mode='soft') for c in coeffs[1])
        
        reconstructed_image = pywt.waverec2(coeffs_thresholded, 'haar')

        # 数据一致性约束
        k_space_reconstructed = np.fft.fft2(reconstructed_image)
        k_space_reconstructed[sparse_k_space == 0] = 0
        reconstructed_image = np.fft.ifft2(k_space_reconstructed).real
        
        # 检查终止条件
        if i > 0 and np.linalg.norm(reconstructed_image - previous_image) < tol:
            print(f'Converged after {i} iterations.')
            break
            
        previous_image = reconstructed_image.copy()
        
    return reconstructed_image


# 初始化存储PSNR和SSIM的列表
psnr_values_ct = []
ssim_values_ct = []
psnr_values_phantom = []
ssim_values_phantom = []

# 对CT图像进行重建并记录PSNR和SSIM
for rate, sparse_k_space in sparse_k_spaces_ct.items():
    reconstructed_ct = ista_reconstruction(sparse_k_space)
    
    # 计算PSNR和SSIM
    psnr_ct = psnr(ct_image, reconstructed_ct)
    ssim_ct = ssim(ct_image, reconstructed_ct, data_range=reconstructed_ct.max() - reconstructed_ct.min())

    # 记录PSNR和SSIM值
    psnr_values_ct.append(psnr_ct)
    ssim_values_ct.append(ssim_ct)

# 对Shepp-Logan phantom进行重建并记录PSNR和SSIM
for rate, sparse_k_space in sparse_k_spaces_phantom.items():
    reconstructed_phantom = ista_reconstruction(sparse_k_space)

    # 计算PSNR和SSIM
    psnr_phantom = psnr(phantom, reconstructed_phantom)
    ssim_phantom = ssim(phantom, reconstructed_phantom, data_range=reconstructed_phantom.max() - reconstructed_phantom.min())

    # 记录PSNR和SSIM值
    psnr_values_phantom.append(psnr_phantom)
    ssim_values_phantom.append(ssim_phantom)

# 绘制PSNR和SSIM随欠采样率变化的曲线
sampling_rates = [0.25, 0.5, 0.75]  # 采样率

plt.figure(figsize=(12, 6))

# PSNR曲线
plt.subplot(1, 2, 1)
plt.plot(sampling_rates, psnr_values_ct, marker='o', label='CT Image')
plt.plot(sampling_rates, psnr_values_phantom, marker='o', label='Phantom Image')
plt.title('PSNR vs. Sampling Rate')
plt.xlabel('Sampling Rate')
plt.ylabel('PSNR (dB)')
plt.legend()
plt.grid()

# SSIM曲线
plt.subplot(1, 2, 2)
plt.plot(sampling_rates, ssim_values_ct, marker='o', label='CT Image')
plt.plot(sampling_rates, ssim_values_phantom, marker='o', label='Phantom Image')
plt.title('SSIM vs. Sampling Rate')
plt.xlabel('Sampling Rate')
plt.ylabel('SSIM')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
# # 对CT图像进行重建并可视化
# for rate, sparse_k_space in sparse_k_spaces_ct.items():
#     reconstructed_ct = ista_reconstruction(sparse_k_space)
    
#     # 计算PSNR和SSIM
#     psnr_ct = psnr(ct_image, reconstructed_ct)
#     ssim_ct = ssim(ct_image, reconstructed_ct, data_range=reconstructed_ct.max() - reconstructed_ct.min())

#     # 计算原始图像的频谱
#     original_k_space = np.fft.fft2(ct_image)
#     original_k_space_magnitude = np.log(np.abs(original_k_space) + 1)

#     # 使用 subplots 创建二行三列的图像
#     fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
#     # 原始CT图像
#     axes[0, 0].imshow(ct_image, cmap='gray')
#     axes[0, 0].set_title("Original CT Image")
#     axes[0, 0].axis('off')

#     # 原始图像的频谱图
#     axes[0, 1].imshow(original_k_space_magnitude, cmap='gray')
#     axes[0, 1].set_title("Original Frequency Spectrum")
#     axes[0, 1].axis('off')

#     # 欠采样后的k空间数据
#     axes[0, 2].imshow(np.log(np.abs(sparse_k_space) + 1), cmap='gray')
#     axes[0, 2].set_title(f"K-space (Rate: {rate*100:.0f}%)")
#     axes[0, 2].axis('off')

#     # 重建图像
#     axes[1, 0].imshow(reconstructed_ct, cmap='gray')
#     axes[1, 0].set_title("Reconstructed CT Image")
#     axes[1, 0].axis('off')

#     # 重建图像的频谱图
#     reconstructed_k_space = np.fft.fft2(reconstructed_ct)
#     reconstructed_k_space_magnitude = np.log(np.abs(reconstructed_k_space) + 1)
#     axes[1, 1].imshow(reconstructed_k_space_magnitude, cmap='gray')
#     axes[1, 1].set_title("Reconstructed Frequency Spectrum")
#     axes[1, 1].axis('off')

#     # PSNR和SSIM显示
#     axes[1, 2].text(0.5, 0.5, f"PSNR: {psnr_ct:.2f} dB\nSSIM: {ssim_ct:.4f}", 
#                     fontsize=12, ha='center', va='center')
#     axes[1, 2].set_title("Metrics")
#     axes[1, 2].axis('off')

#     plt.tight_layout()
#     plt.show()

# # 对Shepp-Logan phantom进行重建并可视化
# for rate, sparse_k_space in sparse_k_spaces_phantom.items():
#     reconstructed_phantom = ista_reconstruction(sparse_k_space)

#     # 计算PSNR和SSIM
#     psnr_phantom = psnr(phantom, reconstructed_phantom)
#     ssim_phantom = ssim(phantom, reconstructed_phantom, data_range=reconstructed_phantom.max() - reconstructed_phantom.min())

#     # 计算原始图像的频谱
#     original_k_space_phantom = np.fft.fft2(phantom)
#     original_k_space_magnitude_phantom = np.log(np.abs(original_k_space_phantom) + 1)

#     # 使用 subplots 创建二行三列的图像
#     fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
#     # 原始Shepp-Logan phantom
#     axes[0, 0].imshow(phantom, cmap='gray')
#     axes[0, 0].set_title("Original Phantom")
#     axes[0, 0].axis('off')

#     # 原始图像的频谱图
#     axes[0, 1].imshow(original_k_space_magnitude_phantom, cmap='gray')
#     axes[0, 1].set_title("Original Phantom Frequency Spectrum")
#     axes[0, 1].axis('off')

#     # 欠采样后的k空间数据
#     axes[0, 2].imshow(np.log(np.abs(sparse_k_space) + 1), cmap='gray')
#     axes[0, 2].set_title(f"K-space (Rate: {rate*100:.0f}%)")
#     axes[0, 2].axis('off')

#     # 重建图像
#     axes[1, 0].imshow(reconstructed_phantom, cmap='gray')
#     axes[1, 0].set_title("Reconstructed Phantom")
#     axes[1, 0].axis('off')

#     # 重建图像的频谱图
#     reconstructed_k_space_phantom = np.fft.fft2(reconstructed_phantom)
#     reconstructed_k_space_magnitude_phantom = np.log(np.abs(reconstructed_k_space_phantom) + 1)
#     axes[1, 1].imshow(reconstructed_k_space_magnitude_phantom, cmap='gray')
#     axes[1, 1].set_title("Reconstructed Phantom Frequency Spectrum")
#     axes[1, 1].axis('off')

#     # PSNR和SSIM显示
#     axes[1, 2].text(0.5, 0.5, f"PSNR: {psnr_phantom:.2f} dB\nSSIM: {ssim_phantom:.4f}", 
#                     fontsize=12, ha='center', va='center')
#     axes[1, 2].set_title("Metrics")
#     axes[1, 2].axis('off')

#     plt.tight_layout()
#     plt.show()


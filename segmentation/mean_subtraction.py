"""
Mean Subtraction 预处理模块
用于生成纹理感知的对比图像，去除亮度影响，突出局部纹理细节
"""

import cv2
import numpy as np
import torch
from mmcv.transforms import Transform
from mmseg.registry import TRANSFORMS


@TRANSFORMS.register_module()
class MeanSubtraction(Transform):
    """
    Mean Subtraction 预处理

    对输入图像减去局部均值，生成纹理增强的对比图像。
    用于阴影检测任务中区分真正的阴影和暗色物体。

    Args:
        kernel_size (int): 高斯滤波核大小，默认15
        sigma (float): 高斯核标准差，0表示自动计算
        normalize (bool): 是否归一化到[0, 255]，默认True
    """

    def __init__(self,
                 kernel_size: int = 15,
                 sigma: float = 0,
                 normalize: bool = True):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.normalize = normalize

    def __call__(self, results):
        """
        对图像应用Mean Subtraction

        Args:
            results (dict): 包含'img'键的字典，值为[H, W, C]的numpy数组

        Returns:
            results (dict): 添加'img_ms'键，包含Mean Subtraction图像
        """
        img = results['img']
        assert isinstance(img, np.ndarray), "Input must be numpy array"
        assert len(img.shape) == 3, f"Expected 3D array [H,W,C], got {img.shape}"

        # 应用Mean Subtraction
        ms_img = self.apply_mean_subtraction(img)

        # 保存到results
        results['img_ms'] = ms_img

        return results

    def apply_mean_subtraction(self, img):
        """
        执行Mean Subtraction操作

        Args:
            img: [H, W, 3] RGB图像，范围[0, 255]

        Returns:
            ms_img: [H, W, 3] Mean Subtraction图像
        """
        # 转换为float32
        img = img.astype(np.float32)

        # 对每个通道分别处理
        ms_channels = []
        for c in range(3):
            channel = img[:, :, c]

            # 计算局部均值（使用高斯滤波，比box filter更平滑）
            local_mean = cv2.GaussianBlur(
                channel,
                (self.kernel_size, self.kernel_size),
                self.sigma
            )

            # 减去均值
            ms_channel = channel - local_mean

            if self.normalize:
                # 归一化到[0, 255]
                ms_min = ms_channel.min()
                ms_max = ms_channel.max()
                if ms_max - ms_min > 1e-8:
                    ms_channel = (ms_channel - ms_min) / (ms_max - ms_min) * 255
                else:
                    ms_channel = np.zeros_like(ms_channel)

            ms_channels.append(ms_channel)

        # 堆叠通道
        ms_img = np.stack(ms_channels, axis=2)

        # 保持原始数据类型
        ms_img = ms_img.astype(img.dtype)

        return ms_img

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"kernel_size={self.kernel_size}, "
                f"sigma={self.sigma}, "
                f"normalize={self.normalize})")


@TRANSFORMS.register_module()
class MultiScaleMeanSubtraction(MeanSubtraction):
    """
    多尺度Mean Subtraction

    使用多个kernel size生成多尺度的纹理特征，可以捕捉不同尺度下的纹理模式。

    Args:
        kernel_sizes (list): 多个高斯滤波核大小，如[7, 15, 31]
        fusion (str): 多尺度特征融合方式，'concat'或'add'
    """

    def __init__(self,
                 kernel_sizes=[7, 15, 31],
                 fusion='concat',
                 **kwargs):
        # 仅使用第一个kernel_size初始化父类
        super().__init__(kernel_size=kernel_sizes[0], **kwargs)
        self.kernel_sizes = kernel_sizes
        self.fusion = fusion

    def apply_mean_subtraction(self, img):
        """
        应用多尺度Mean Subtraction

        Args:
            img: [H, W, 3] RGB图像

        Returns:
            ms_img: [H, W, 3*C] 或 [H, W, 3] 多尺度融合图像
        """
        img = img.astype(np.float32)

        # 对每个尺度计算Mean Subtraction
        multi_scale_features = []
        for kernel_size in self.kernel_sizes:
            self.kernel_size = kernel_size  # 临时修改kernel_size
            ms_img = super().apply_mean_subtraction(img)
            multi_scale_features.append(ms_img)

        # 融合多尺度特征
        if self.fusion == 'concat':
            # 拼接：[H, W, 3] * C -> [H, W, 3*C]
            ms_img = np.concatenate(multi_scale_features, axis=2)
        elif self.fusion == 'add':
            # 加权平均
            ms_img = np.mean(multi_scale_features, axis=0)
        else:
            raise ValueError(f"Unknown fusion mode: {self.fusion}")

        return ms_img.astype(img.dtype)


def compute_mean_subtraction_torch(img_tensor, kernel_size=15, sigma=0):
    """
    PyTorch版本的Mean Subtraction（用于推理时）

    Args:
        img_tensor: [B, 3, H, W] RGB图像张量，范围[0, 1]
        kernel_size (int): 高斯核大小
        sigma (float): 高斯核标准差

    Returns:
        ms_tensor: [B, 3, H, W] Mean Subtraction张量
    """
    import torchvision
    from torchvision.transforms import GaussianBlur

    assert len(img_tensor.shape) == 4, f"Expected [B,C,H,W], got {img_tensor.shape}"

    # 高斯模糊计算局部均值
    gaussian = GaussianBlur(kernel_size=kernel_size, sigma=sigma)
    local_mean = gaussian(img_tensor)

    # 减去均值
    ms_tensor = img_tensor - local_mean

    # 归一化到[0, 1]
    ms_min = ms_tensor.amin(dim=(2, 3), keepdim=True)
    ms_max = ms_tensor.amax(dim=(2, 3), keepdim=True)
    ms_tensor = (ms_tensor - ms_min) / (ms_max - ms_min + 1e-8)

    return ms_tensor


# 测试代码
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # 读取测试图像
    img_path = "test_image.jpg"  # 替换为实际路径
    img = cv2.imread(img_path)
    if img is None:
        # 创建一个测试图像
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 应用Mean Subtraction
    ms_transform = MeanSubtraction(kernel_size=15)
    ms_img = ms_transform.apply_mean_subtraction(img)

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(ms_img.astype(np.uint8))
    axes[1].set_title('Mean Subtraction')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig('mean_subtraction_demo.png')
    print("Visualization saved to 'mean_subtraction_demo.png'")

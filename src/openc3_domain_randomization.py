from typing import Dict, Union
import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms_v2


class VisualDomainRandomizer(nn.Module):
    """用于具身智能策略训练的视觉领域随机化（DR）模块。

    针对 observation.images.top 与 observation.images.wrist 进行时序一致的数据增强：
    1. ColorJitter: 模拟环境光照强度、色温与对比度漂移
    2. GaussianBlur: 模拟相机对焦模糊与运动拖影
    3. GaussianNoise: 模拟图像传感器热噪声与低照度噪点
    4. RandomErasing: 模拟推运过程中的局部阴影与短暂视觉遮挡
    """

    def __init__(
        self,
        brightness: float = 0.25,
        contrast: float = 0.25,
        saturation: float = 0.2,
        hue: float = 0.05,
        noise_std: float = 0.02,
        blur_prob: float = 0.3,
        erase_prob: float = 0.2,
    ):
        super().__init__()
        self.noise_std = noise_std
        self.blur_prob = blur_prob
        self.erase_prob = erase_prob

        # 色彩与光照抖动
        self.color_jitter = transforms_v2.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )

        # 高斯模糊算子
        self.gaussian_blur = transforms_v2.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.5))

        # 随机擦除 (模拟局部盲区/遮挡)
        self.random_erase = transforms_v2.RandomErasing(
            p=1.0,
            scale=(0.02, 0.12),
            ratio=(0.5, 2.0),
            value="random",
        )

    def _apply_transform_to_sequence(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """对输入张量施加增强。

        支持输入形状:
        - 单帧: (C, H, W)
        - 时序批次: (B, T, C, H, W)
        """
        is_batched = img_tensor.ndim == 5
        if not is_batched:
            # (C, H, W) -> (1, 1, C, H, W)
            orig_shape = img_tensor.shape
            img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)

        B, T, C, H, W = img_tensor.shape
        # 将时序帧拼接在通道或 Batch 维处理，保证时序参数一致性
        flattened = img_tensor.view(B * T, C, H, W)

        # 1. 色彩抖动变换
        augmented = self.color_jitter(flattened)

        # 2. 概率性高斯模糊
        if torch.rand(1).item() < self.blur_prob:
            augmented = self.gaussian_blur(augmented)

        # 3. 注入高斯感光噪声
        if self.noise_std > 0.0:
            noise = torch.randn_like(augmented) * self.noise_std
            augmented = augmented + noise

        # 4. 概率性局部擦除遮挡
        if torch.rand(1).item() < self.erase_prob:
            augmented = self.random_erase(augmented)

        # 严格截断至合法图像区间 [0.0, 1.0]
        augmented = torch.clamp(augmented, 0.0, 1.0)
        augmented = augmented.view(B, T, C, H, W)

        if not is_batched:
            return augmented.squeeze(0).squeeze(0)
        return augmented

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """对 LeRobot 批次字典中的所有图像模态施加领域随机化。"""
        out_batch = dict(batch)
        for key in ["observation.images.top", "observation.images.wrist"]:
            if key in out_batch and isinstance(out_batch[key], torch.Tensor):
                out_batch[key] = self._apply_transform_to_sequence(out_batch[key])
        return out_batch
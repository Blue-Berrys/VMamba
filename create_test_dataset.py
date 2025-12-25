#!/usr/bin/env python3
"""
创建临时测试数据集
用于验证模型是否正常工作
"""

import os
import shutil
from PIL import Image
import numpy as np

def create_dummy_sbu_dataset():
    """创建虚拟SBU数据集用于测试"""

    print("=" * 60)
    print("创建临时测试数据集")
    print("=" * 60)

    # 数据集路径
    data_dir = "data/sbu"
    img_dir = os.path.join(data_dir, "img")
    label_dir = os.path.join(data_dir, "label")

    # 创建目录
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    print(f"\n创建目录:")
    print(f"  {img_dir}")
    print(f"  {label_dir}")

    # 创建20张测试图像
    num_samples = 20
    print(f"\n生成 {num_samples} 张测试图像...")

    for i in range(1, num_samples + 1):
        # 生成随机图像 (640x480 RGB)
        img_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(os.path.join(img_dir, f"{i:04d}.jpg"))

        # 生成随机标注 (二值化: 0或255)
        label_array = np.random.choice([0, 255], size=(480, 640)).astype(np.uint8)
        label = Image.fromarray(label_array, mode='L')
        label.save(os.path.join(label_dir, f"{i:04d}.png"))

    print(f"✓ 成功创建 {num_samples} 张图像和标注")

    # 创建README
    readme_path = os.path.join(data_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write("# SBU测试数据集\n\n")
        f.write(f"这是一个临时生成的测试数据集，用于验证模型配置是否正确。\n\n")
        f.write(f"图像数量: {num_samples}\n")
        f.write(f"图像尺寸: 640x480\n")
        f.write(f"标注类型: 二值化 (0=非阴影, 255=阴影)\n\n")
        f.write("⚠ 警告: 这不是真实的SBU数据集！仅用于测试！\n")
        f.write("\n请替换为真实的SBU数据集后再进行正式训练。\n")

    print(f"\n✓ 创建说明文件: {readme_path}")

    print("\n" + "=" * 60)
    print("✅ 临时数据集创建成功！")
    print("=" * 60)
    print(f"\n数据集位置: {data_dir}")
    print(f"  - 图像: {img_dir}")
    print(f"  - 标注: {label_dir}")
    print(f"  - 说明: {readme_path}")
    print(f"\n现在可以开始训练测试了！")
    print(f"\n启动命令:")
    print(f"  cd segmentation")
    print(f"  bash tools/dist_train.sh configs/sbu/sbu_shadow_dual_stream_base_40k.py 4")
    print(f"\n⚠ 注意: 使用真实SBU数据集以获得有意义的训练结果！")

if __name__ == "__main__":
    import sys
    os.chdir("/root/autodl-tmp/code/VMamba")
    create_dummy_sbu_dataset()

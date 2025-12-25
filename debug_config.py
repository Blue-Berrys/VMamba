#!/usr/bin/env python3
"""
逐步测试配置文件加载
帮助定位问题
"""

import sys
import os

print("=" * 60)
print("配置文件加载测试")
print("=" * 60)

# 设置路径
sys.path.insert(0, "/root/autodl-tmp/code/VMamba")
os.chdir("/root/autodl-tmp/code/VMamba")

print(f"工作目录: {os.getcwd()}")
print(f"Python路径: {sys.path[:3]}")
print()

# 步骤1: 测试基础导入
print("步骤1: 测试基础导入...")
print("-" * 60)
try:
    from mmengine import Config
    print("✓ mmengine.Config 导入成功")
except Exception as e:
    print(f"✗ mmengine.Config 导入失败: {e}")
    sys.exit(1)

# 步骤2: 测试vmamba导入
print()
print("步骤2: 测试vmamba导入...")
print("-" * 60)
try:
    sys.path.insert(0, "classification/models")
    import vmamba
    print("✓ vmamba 导入成功")
except Exception as e:
    print(f"✗ vmamba 导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 步骤3: 测试vmamba_dual导入
print()
print("步骤3: 测试vmamba_dual导入...")
print("-" * 60)
try:
    import vmamba_dual
    print("✓ vmamba_dual 导入成功")
except Exception as e:
    print(f"✗ vmamba_dual 导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 步骤4: 测试模型注册
print()
print("步骤4: 测试模型注册...")
print("-" * 60)
try:
    # 添加segmentation到路径
    sys.path.insert(0, "segmentation")
    from model import MM_DualStreamVSSM, DUAL_STREAM_AVAILABLE
    print(f"✓ model.py 导入成功")
    print(f"  DUAL_STREAM_AVAILABLE = {DUAL_STREAM_AVAILABLE}")
except Exception as e:
    print(f"✗ model.py 导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 步骤5: 测试配置文件加载
print()
print("步骤5: 测试配置文件加载...")
print("-" * 60)
try:
    config_path = "segmentation/configs/sbu/sbu_shadow_dual_stream_base_40k.py"
    cfg = Config.fromfile(config_path)
    print("✓ 配置文件加载成功")
    print(f"  模型类型: {cfg.model.type}")
    print(f"  Backbone类型: {cfg.model.backbone.type}")
except Exception as e:
    print(f"✗ 配置文件加载失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 步骤6: 测试模型构建
print()
print("步骤6: 测试模型构建...")
print("-" * 60)
try:
    from mmseg.registry import MODELS

    # 构建backbone
    backbone = MODELS.build(cfg.model.backbone)
    print(f"✓ Backbone构建成功: {type(backbone).__name__}")

    # 测试前向传播
    import torch
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        outputs = backbone(x)
    print(f"✓ 前向传播成功，输出 {len(outputs)} 个特征图")

    # 打印输出形状
    for i, out in enumerate(outputs):
        print(f"  Stage {i}: {out.shape}")

except Exception as e:
    print(f"✗ 模型构建失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 总结
print()
print("=" * 60)
print("✅ 所有测试通过！")
print("=" * 60)
print()
print("可以开始训练了：")
print("  cd segmentation")
print("  bash tools/dist_train.sh configs/sbu/sbu_shadow_dual_stream_base_40k.py 4")
print()

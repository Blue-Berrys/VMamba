#!/usr/bin/env python3
"""
双流VMamba完整诊断脚本
用于检测和解决导入问题
"""

import sys
import os
import traceback

def test_import_vmamba():
    """测试基础vmamba模块导入"""
    print("=" * 60)
    print("测试 1: 导入基础vmamba模块")
    print("=" * 60)

    try:
        # 添加路径
        models_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "classification/models")
        if models_path not in sys.path:
            sys.path.insert(0, models_path)

        # 导入vmamba
        import vmamba
        print("✓ 成功导入 vmamba 模块")

        # 检查关键类
        classes = ['SS2D', 'VSSBlock', 'VSSM', 'Backbone_VSSM']
        for cls in classes:
            if hasattr(vmamba, cls):
                print(f"  ✓ {cls} 可用")
            else:
                print(f"  ✗ {cls} 不可用")
                return False

        return True

    except Exception as e:
        print(f"✗ 导入失败: {e}")
        traceback.print_exc()
        return False

def test_import_vmamba_dual():
    """测试vmamba_dual模块导入"""
    print("\n" + "=" * 60)
    print("测试 2: 导入vmamba_dual模块")
    print("=" * 60)

    try:
        # 添加路径
        models_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "classification/models")
        if models_path not in sys.path:
            sys.path.insert(0, models_path)

        # 导入vmamba_dual
        import vmamba_dual
        print("✓ 成功导入 vmamba_dual 模块")

        # 检查关键类
        classes = ['DualStreamVSSBlock', 'DualStreamVSSM', 'Backbone_DualStreamVSSM']
        for cls in classes:
            if hasattr(vmamba_dual, cls):
                print(f"  ✓ {cls} 可用")
            else:
                print(f"  ✗ {cls} 不可用")
                return False

        return True

    except Exception as e:
        print(f"✗ 导入失败: {e}")
        traceback.print_exc()
        return False

def test_create_model():
    """测试创建模型"""
    print("\n" + "=" * 60)
    print("测试 3: 创建双流模型")
    print("=" * 60)

    try:
        import torch
        from vmamba_dual import Backbone_DualStreamVSSM

        # 创建小模型测试
        model = Backbone_DualStreamVSSM(
            depths=[1, 1, 1, 1],  # 最小配置
            dims_s1=[64, 128, 256, 512],
            dims_s2=[32, 64, 128, 256],
            drop_path_rate=0.1,
        )
        print("✓ 成功创建模型")

        # 测试前向传播
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            outputs = model(x)
        print(f"✓ 前向传播成功，输出 {len(outputs)} 个特征图")

        # 打印输出形状
        for i, out in enumerate(outputs):
            print(f"  Stage {i}: {out.shape}")

        return True

    except Exception as e:
        print(f"✗ 模型创建或前向传播失败: {e}")
        traceback.print_exc()
        return False

def test_mean_subtraction():
    """测试Mean Subtraction"""
    print("\n" + "=" * 60)
    print("测试 4: Mean Subtraction功能")
    print("=" * 60)

    try:
        import torch
        from vmamba_dual import Backbone_DualStreamVSSM

        model = Backbone_DualStreamVSSM(
            depths=[1, 1, 1, 1],
            dims_s1=[64, 128, 256, 512],
            dims_s2=[32, 64, 128, 256],
            drop_path_rate=0.1,
        )

        # 创建测试图像
        x = torch.randn(1, 3, 224, 224) * 255

        # 测试Mean Subtraction
        with torch.no_grad():
            x_ms = model._compute_mean_subtraction(x)

        print(f"✓ Mean Subtraction成功")
        print(f"  输入范围: [{x.min():.2f}, {x.max():.2f}]")
        print(f"  输出范围: [{x_ms.min():.2f}, {x_ms.max():.2f}]")

        return True

    except Exception as e:
        print(f"✗ Mean Subtraction失败: {e}")
        traceback.print_exc()
        return False

def test_dependencies():
    """测试依赖包"""
    print("\n" + "=" * 60)
    print("测试 5: 检查依赖包")
    print("=" * 60)

    required = {
        'torch': 'PyTorch',
        'timm': 'timm',
        'mmseg': 'MMSegmentation',
        'mmengine': 'MMEngine',
    }

    all_ok = True
    for module, name in required.items():
        try:
            __import__(module)
            print(f"  ✓ {name} ({module})")
        except ImportError:
            print(f"  ✗ {name} ({module}) 未安装")
            all_ok = False

    return all_ok

def main():
    """运行所有测试"""
    print("\n双流VMamba诊断工具")
    print("=" * 60)

    # 检查当前目录
    print(f"当前目录: {os.getcwd()}")
    print(f"Python版本: {sys.version}")

    results = []

    # 运行测试
    results.append(("依赖检查", test_dependencies()))
    results.append(("基础vmamba导入", test_import_vmamba()))
    results.append(("vmamba_dual导入", test_import_vmamba_dual()))
    results.append(("模型创建", test_create_model()))
    results.append(("Mean Subtraction", test_mean_subtraction()))

    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n🎉 所有测试通过！可以开始训练。")
        return 0
    else:
        print("\n❌ 部分测试失败，请检查上述错误信息。")
        return 1

if __name__ == "__main__":
    sys.exit(main())

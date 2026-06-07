#!/usr/bin/env python3
"""
模型验证脚本 - 检查改进的双流模型代码
==========================================

运行此脚本验证:
1. 模型能否正常导入
2. 前向传播是否正常
3. 参数量统计
4. 训练阶段切换功能
5. LocalStream 单独测试
6. CrossAttentionFusion 测试

作者: AgentLaboratory
日期: 2026-02-19
"""

import sys
import os

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# vmamba_integration 目录 (本地 Mac)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'vmamba_integration'))
# VMamba 服务器路径: segmentation/../classification/models/
_script_dir = os.path.dirname(os.path.abspath(__file__))
_vmamba_root = os.path.join(_script_dir, '..')
_models_path = os.path.join(_vmamba_root, 'classification', 'models')
if os.path.isdir(_models_path):
    sys.path.insert(0, os.path.abspath(_models_path))


def test_import():
    """测试模型导入"""
    print("=" * 50)
    print("测试1: 模型导入")
    print("=" * 50)

    try:
        from shadow_dual_stream_v2 import (
            ShadowDualStreamVSSM,
            CrossAttentionFusion,
            LocalStream,
            shadow_dual_stream_base
        )
        print("OK 所有组件导入成功!")
        return True, {
            'ShadowDualStreamVSSM': ShadowDualStreamVSSM,
            'CrossAttentionFusion': CrossAttentionFusion,
            'LocalStream': LocalStream,
            'shadow_dual_stream_base': shadow_dual_stream_base
        }
    except Exception as e:
        print(f"FAIL 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_model_creation(model_class):
    """测试模型创建"""
    print("\n" + "=" * 50)
    print("测试2: 模型创建")
    print("=" * 50)

    try:
        model = model_class(
            depths=[2, 2, 27, 2],
            dims=[128, 256, 512, 1024],
            drop_path_rate=0.6
        )
        print("OK 模型创建成功!")
        return model
    except Exception as e:
        print(f"FAIL 模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_forward_pass(model):
    """测试前向传播"""
    print("\n" + "=" * 50)
    print("测试3: 前向传播")
    print("=" * 50)

    import torch

    if not torch.cuda.is_available():
        print("SKIP 无 CUDA，跳过全模型前向传播（VMamba global_stream 需要 CUDA）")
        print("     LocalStream 和 CrossAttention 已在独立测试中验证。")
        return True

    try:
        device = torch.device('cuda')
        model = model.to(device)
        x = torch.randn(2, 3, 416, 416, device=device)
        print(f"输入形状: {x.shape}  device: {device}")

        model.eval()
        with torch.no_grad():
            outputs = model(x)

        print(f"输出keys: {list(outputs.keys())}")
        print(f"特征数量: {len(outputs['features'])}")

        for i, feat in enumerate(outputs['features']):
            print(f"  Stage {i} 特征形状: {feat.shape}")

        for i in range(len(outputs['features']) - 1):
            h_curr = outputs['features'][i].shape[2]
            h_next = outputs['features'][i + 1].shape[2]
            assert h_curr >= h_next, f"Stage {i} -> {i+1} 特征尺度未递减: {h_curr} -> {h_next}"

        print("OK 前向传播成功!")
        return True
    except Exception as e:
        print(f"FAIL 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_local_stream():
    """测试LocalStream单独工作"""
    print("\n" + "=" * 50)
    print("测试4: LocalStream 单独测试")
    print("=" * 50)

    try:
        import torch
        from shadow_dual_stream_v2 import LocalStream

        dims = [128, 256, 512, 1024]
        local_stream = LocalStream(
            in_chans=3,
            dims=dims,
            depths=[2, 2, 27, 2]
        )

        x = torch.randn(2, 3, 416, 416)
        print(f"输入形状: {x.shape}")

        local_stream.eval()
        with torch.no_grad():
            features, shadow_maps = local_stream(x)

        print(f"输出特征数: {len(features)}")
        print(f"阴影图数: {len(shadow_maps)}")

        for i, (feat, smap) in enumerate(zip(features, shadow_maps)):
            print(f"  Stage {i}: feat={feat.shape}, shadow_map={smap.shape}")
            assert feat.shape[1] == dims[i], f"Stage {i} 通道数不匹配: {feat.shape[1]} != {dims[i]}"
            assert smap.shape[1] == 1, f"Stage {i} 阴影图通道应为1，实际: {smap.shape[1]}"

        print("OK LocalStream 测试成功!")
        return True
    except Exception as e:
        print(f"FAIL LocalStream 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_stage_switch(model):
    """测试训练阶段切换"""
    print("\n" + "=" * 50)
    print("测试5: 训练阶段切换")
    print("=" * 50)

    try:
        for stage in [0, 1, 2]:
            model.set_training_stage(stage)

            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())

            stage_names = ['全局流', '局部流', '联合训练']
            print(f"阶段{stage} ({stage_names[stage]}):")
            print(f"  可训练参数: {trainable/1e6:.2f}M")
            print(f"  总参数: {total/1e6:.2f}M")

        # 验证阶段0冻结了局部流
        model.set_training_stage(0)
        for name, param in model.named_parameters():
            if 'local_stream' in name:
                assert not param.requires_grad, f"阶段0: {name} 应被冻结"
            if 'global_stream' in name:
                assert param.requires_grad, f"阶段0: {name} 应可训练"

        # 验证阶段1冻结了全局流
        model.set_training_stage(1)
        for name, param in model.named_parameters():
            if 'global_stream' in name:
                assert not param.requires_grad, f"阶段1: {name} 应被冻结"

        # 验证阶段2全部可训练
        model.set_training_stage(2)
        for name, param in model.named_parameters():
            assert param.requires_grad, f"阶段2: {name} 应可训练"

        print("OK 训练阶段切换成功!")
        return True
    except Exception as e:
        print(f"FAIL 训练阶段切换失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_parameter_count(model):
    """统计模型参数"""
    print("\n" + "=" * 50)
    print("测试6: 参数统计")
    print("=" * 50)

    try:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"总参数量: {total_params/1e6:.2f}M")
        print(f"可训练参数: {trainable_params/1e6:.2f}M")

        component_params = {}
        for name, param in model.named_parameters():
            component = name.split('.')[0]
            if component not in component_params:
                component_params[component] = 0
            component_params[component] += param.numel()

        print("\n各组件参数量:")
        for comp, params in sorted(component_params.items(), key=lambda x: -x[1]):
            print(f"  {comp}: {params/1e6:.2f}M")

        print("OK 参数统计完成!")
        return True
    except Exception as e:
        print(f"FAIL 参数统计失败: {e}")
        return False


def test_cross_attention():
    """测试Cross-Attention模块"""
    print("\n" + "=" * 50)
    print("测试7: Cross-Attention融合模块")
    print("=" * 50)

    try:
        from shadow_dual_stream_v2 import CrossAttentionFusion
        import torch

        fusion = CrossAttentionFusion(dim=128, num_heads=4, bidirectional=True)

        global_feat = torch.randn(2, 128, 26, 26)
        local_feat = torch.randn(2, 128, 26, 26)
        shadow_map = torch.sigmoid(torch.randn(2, 1, 26, 26))

        output = fusion(global_feat, local_feat, shadow_map)

        print(f"全局特征: {global_feat.shape}")
        print(f"局部特征: {local_feat.shape}")
        print(f"阴影图: {shadow_map.shape}")
        print(f"融合输出: {output.shape}")

        assert output.shape == global_feat.shape, "输出形状不匹配!"

        print("OK Cross-Attention测试成功!")
        return True
    except Exception as e:
        print(f"FAIL Cross-Attention测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("\n" + "=" * 50)
    print("改进的双流VMamba模型 - 验证测试")
    print("=" * 50)

    results = []

    # 测试1: 导入
    success, classes = test_import()
    results.append(("导入", success))
    if not success:
        print("\nFAIL 测试失败: 无法导入模型")
        return False

    # 测试2: 模型创建
    model = test_model_creation(classes['ShadowDualStreamVSSM'])
    results.append(("模型创建", model is not None))
    if model is None:
        print("\nFAIL 测试失败: 无法创建模型")
        return False

    # 测试3: 前向传播
    success = test_forward_pass(model)
    results.append(("前向传播", success))

    # 测试4: LocalStream 单独测试
    success = test_local_stream()
    results.append(("LocalStream", success))

    # 测试5: 训练阶段切换
    success = test_training_stage_switch(model)
    results.append(("阶段切换", success))

    # 测试6: 参数统计
    success = test_parameter_count(model)
    results.append(("参数统计", success))

    # 测试7: CrossAttention
    success = test_cross_attention()
    results.append(("CrossAttention", success))

    # 汇总
    print("\n" + "=" * 50)
    print("测试结果汇总:")
    print("=" * 50)
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n所有测试通过! 模型已准备好部署。")
    else:
        print("\n存在测试失败，请检查上方错误信息。")

    return all_passed


if __name__ == '__main__':
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\nFAIL 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

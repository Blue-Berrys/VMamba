"""
测试双流VMamba模块导入
"""
import sys
import os

# 添加classification/models到路径
vmamba_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "classification/models")
sys.path.insert(0, vmamba_path)

try:
    import vmamba_dual
    print("✓ Successfully imported vmamba_dual module")

    # 检查可用的类
    classes = [name for name in dir(vmamba_dual) if not name.startswith('_') and name[0].isupper()]
    print(f"✓ Available classes: {classes}")

    # 检查关键类
    if hasattr(vmamba_dual, 'DualStreamVSSBlock'):
        print("✓ DualStreamVSSBlock found")
    if hasattr(vmamba_dual, 'DualStreamVSSM'):
        print("✓ DualStreamVSSM found")
    if hasattr(vmamba_dual, 'Backbone_DualStreamVSSM'):
        print("✓ Backbone_DualStreamVSSM found")

    print("\n✓ All imports successful!")

except Exception as e:
    print(f"✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

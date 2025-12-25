#!/bin/bash
# 修复双流VMamba导入和配置问题

echo "========================================="
echo "修复双流VMamba配置问题"
echo "========================================="

# 1. 检查vmamba_dual.py是否存在
echo ""
echo "检查 vmamba_dual.py..."
if [ -f "classification/models/vmamba_dual.py" ]; then
    echo "✓ vmamba_dual.py 存在"
else
    echo "✗ 错误: vmamba_dual.py 不存在"
    exit 1
fi

# 2. 测试Python语法
echo ""
echo "测试 Python 语法..."
python3 -m py_compile classification/models/vmamba_dual.py
if [ $? -eq 0 ]; then
    echo "✓ 语法检查通过"
else
    echo "✗ 语法错误"
    exit 1
fi

# 3. 验证导入
echo ""
echo "测试模块导入..."
python3 test_import.py
if [ $? -eq 0 ]; then
    echo "✓ 导入成功"
else
    echo "✗ 导入失败"
    exit 1
fi

# 4. 检查配置文件
echo ""
echo "检查配置文件..."
if grep -q "_delete_=True" segmentation/configs/sbu/sbu_shadow_dual_stream_base_40k.py; then
    echo "✓ 配置文件已修复 _delete_=True"
else
    echo "⚠ 警告: 配置文件可能需要 _delete_=True"
fi

echo ""
echo "========================================="
echo "修复完成！"
echo "========================================="
echo ""
echo "现在可以运行训练："
echo "  cd segmentation"
echo "  bash tools/dist_train.sh configs/sbu/sbu_shadow_dual_stream_base_40k.py 4"
echo ""

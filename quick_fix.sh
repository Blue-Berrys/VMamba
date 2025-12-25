#!/bin/bash
# 快速修复脚本 - 在服务器上运行

set -e  # 遇到错误立即退出

echo "=========================================="
echo "双流VMamba快速修复脚本"
echo "=========================================="

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 函数：打印成功信息
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# 函数：打印错误信息
print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# 函数：打印警告信息
print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# 1. 检查文件
echo ""
echo "步骤 1: 检查必需文件..."
FILES=(
    "classification/models/vmamba.py"
    "classification/models/vmamba_dual.py"
    "segmentation/model.py"
    "segmentation/configs/sbu/sbu_shadow_dual_stream_base_40k.py"
)

for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        print_success "$file 存在"
    else
        print_error "$file 不存在"
        exit 1
    fi
done

# 2. 测试Python语法
echo ""
echo "步骤 2: 测试Python语法..."
python3 -m py_compile classification/models/vmamba_dual.py
if [ $? -eq 0 ]; then
    print_success "vmamba_dual.py 语法检查通过"
else
    print_error "vmamba_dual.py 语法错误"
    exit 1
fi

# 3. 检查vmamba_dual.py的导入
echo ""
echo "步骤 3: 检查vmamba_dual.py导入..."

# 创建临时测试脚本
cat > /tmp/test_import_vmamba_dual.py << 'EOF'
import sys
import os

# 添加路径
sys.path.insert(0, "classification/models")

try:
    import vmamba
    print("✓ vmamba 导入成功")
except Exception as e:
    print(f"✗ vmamba 导入失败: {e}")
    sys.exit(1)

try:
    import vmamba_dual
    print("✓ vmamba_dual 导入成功")
except Exception as e:
    print(f"✗ vmamba_dual 导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 检查类
from vmamba_dual import Backbone_DualStreamVSSM
print("✓ Backbone_DualStreamVSSM 类可用")

print("\n所有导入测试通过！")
EOF

python3 /tmp/test_import_vmamba_dual.py
if [ $? -eq 0 ]; then
    print ""
    print_success "导入测试通过"
else
    print ""
    print_error "导入测试失败"
    echo ""
    echo "请尝试以下修复："
    echo "1. 检查vmamba.py和vmamba_dual.py在同一目录"
    echo "2. 检查vmamba.py的导入是否正常"
    echo "3. 运行: python3 test_dual_stream.py"
    exit 1
fi

# 4. 验证配置文件
echo ""
echo "步骤 4: 验证配置文件..."
if grep -q "_delete_=True" segmentation/configs/sbu/sbu_shadow_dual_stream_base_40k.py; then
    print_success "配置文件包含 _delete_=True"
else
    print_warning "配置文件可能缺少 _delete_=True"
    echo "建议：在optim_wrapper配置中添加 _delete_=True"
fi

# 5. 测试模型创建
echo ""
echo "步骤 5: 测试模型创建..."
cat > /tmp/test_model_creation.py << 'EOF'
import sys
sys.path.insert(0, "classification/models")

import torch
from vmamba_dual import Backbone_DualStreamVSSM

print("创建小模型测试...")
model = Backbone_DualStreamVSSM(
    depths=[1, 1, 1, 1],
    dims_s1=[64, 128, 256, 512],
    dims_s2=[32, 64, 128, 256],
    drop_path_rate=0.1,
)
print("✓ 模型创建成功")

print("测试前向传播...")
x = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    outputs = model(x)
print(f"✓ 前向传播成功，输出 {len(outputs)} 个特征图")

print("测试Mean Subtraction...")
x = torch.randn(1, 3, 224, 224) * 255
x_ms = model._compute_mean_subtraction(x)
print(f"✓ Mean Subtraction成功")

print("\n所有模型测试通过！")
EOF

python3 /tmp/test_model_creation.py
if [ $? -eq 0 ]; then
    echo ""
    print_success "模型测试通过"
else
    echo ""
    print_error "模型测试失败"
    exit 1
fi

# 6. 总结
echo ""
echo "=========================================="
echo "修复总结"
echo "=========================================="
echo ""
print_success "所有检查通过！"
echo ""
echo "现在可以开始训练："
echo ""
echo "  cd segmentation"
echo "  bash tools/dist_train.sh configs/sbu/sbu_shadow_dual_stream_base_40k.py 4"
echo ""
echo "或运行完整测试："
echo ""
echo "  python3 test_dual_stream.py"
echo ""
echo "=========================================="

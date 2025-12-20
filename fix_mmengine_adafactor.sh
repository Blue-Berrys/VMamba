#!/bin/bash
# 修复 mmengine Adafactor 重复注册问题

echo "查找 mmengine 安装路径..."
MMENGINE_PATH=$(python -c "import mmengine; import os; print(os.path.dirname(mmengine.__file__))" 2>/dev/null)

if [ -z "$MMENGINE_PATH" ]; then
    echo "错误: 未找到 mmengine，请确保已激活 conda 环境"
    exit 1
fi

BUILDER_FILE="$MMENGINE_PATH/optim/optimizer/builder.py"

if [ ! -f "$BUILDER_FILE" ]; then
    echo "错误: 未找到文件 $BUILDER_FILE"
    exit 1
fi

echo "找到 mmengine: $MMENGINE_PATH"
echo "修复文件: $BUILDER_FILE"

# 备份
if [ ! -f "${BUILDER_FILE}.backup" ]; then
    cp "$BUILDER_FILE" "${BUILDER_FILE}.backup"
    echo "已备份: ${BUILDER_FILE}.backup"
fi

# 使用 Python 修复
python << PYTHON_SCRIPT
import re

file_path = "$BUILDER_FILE"

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 检查是否已修复
if '# Fixed: PyTorch 2.9+' in content:
    print("文件已经修复过")
    exit(0)

# 查找并替换 Adafactor 注册行
# 模式1: OPTIMIZERS.register_module(name='Adafactor', module=Adafactor)
pattern1 = r"(OPTIMIZERS\.register_module\(name=['\"]Adafactor['\"].*?\))"
replacement1 = r"""try:
        \1
    except KeyError:
        # Fixed: PyTorch 2.9+ compatibility - Adafactor already registered
        pass"""

# 先尝试精确匹配
if "OPTIMIZERS.register_module(name='Adafactor'" in content or 'OPTIMIZERS.register_module(name="Adafactor"' in content:
    # 找到包含 Adafactor 注册的行
    lines = content.split('\n')
    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if ('OPTIMIZERS.register_module' in line and 'Adafactor' in line and 
            'try:' not in ''.join(lines[max(0, i-3):i])):
            indent = len(line) - len(line.lstrip())
            new_lines.append(' ' * indent + 'try:')
            new_lines.append(' ' * (indent + 4) + '# Fixed: PyTorch 2.9+ compatibility')
            new_lines.append(' ' * (indent + 4) + line.lstrip())
            i += 1
            # 添加 except
            if i < len(lines) and lines[i].strip() == '':
                new_lines.append(lines[i])
                i += 1
            new_lines.append(' ' * indent + 'except KeyError:')
            new_lines.append(' ' * (indent + 4) + 'pass  # Adafactor already registered in torch.optim')
        else:
            new_lines.append(line)
        i += 1
    
    new_content = '\n'.join(new_lines)
    
    if new_content != content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print("修复完成!")
    else:
        print("未找到需要修复的代码")
else:
    print("未找到 Adafactor 注册代码")
PYTHON_SCRIPT

echo ""
echo "修复完成！测试:"
python -c "from mmengine.runner import Runner; print('✓ mmengine 导入成功')" 2>&1


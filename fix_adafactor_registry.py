#!/usr/bin/env python
"""
修复 mmengine 0.10.1 与 PyTorch 2.9+ 的 Adafactor 重复注册问题

使用方法（在激活 conda 环境后）:
    python fix_adafactor_registry.py
"""

import os
import sys
import shutil
from pathlib import Path

def find_mmengine_path():
    """查找 mmengine 安装路径"""
    try:
        import mmengine
        mmengine_path = Path(mmengine.__file__).parent
        print(f"找到 mmengine 路径: {mmengine_path}")
        return mmengine_path
    except ImportError:
        print("错误: 未找到 mmengine，请先安装 mmengine")
        print("请确保已激活 conda 环境并安装 mmengine==0.10.1")
        sys.exit(1)

def backup_file(file_path):
    """备份文件"""
    backup_path = file_path.with_suffix(file_path.suffix + '.backup')
    if not backup_path.exists():
        shutil.copy2(file_path, backup_path)
        print(f"✓ 已备份文件: {backup_path}")
    else:
        print(f" 备份文件已存在: {backup_path}")
    return backup_path

def fix_adafactor_registry():
    """修复 Adafactor 重复注册问题"""
    mmengine_path = find_mmengine_path()
    builder_file = mmengine_path / 'optim' / 'optimizer' / 'builder.py'
    
    if not builder_file.exists():
        print(f"错误: 未找到文件 {builder_file}")
        sys.exit(1)
    
    print(f"修复文件: {builder_file}")
    
    # 备份原文件
    backup_file(builder_file)
    
    # 读取文件内容
    with open(builder_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 检查是否已经修复过
    content_str = ''.join(lines)
    if '# Fixed: PyTorch 2.9+ compatibility' in content_str:
        print("文件已经修复过，跳过")
        return True
    
    # 查找 register_transformers_optimizers 函数
    new_lines = []
    in_function = False
    function_indent = 0
    fixed = False
    
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        
        # 检测函数开始
        if 'def register_transformers_optimizers():' in line:
            in_function = True
            function_indent = indent
            new_lines.append(line)
            i += 1
            continue
        
        # 在函数内部查找 Adafactor 注册
        if in_function:
            # 检查是否是函数结束（下一个 def 或类定义，且缩进小于等于函数缩进）
            if stripped.startswith('def ') or stripped.startswith('class '):
                if indent <= function_indent:
                    in_function = False
            
            # 查找 OPTIMIZERS.register_module(name='Adafactor' 或 name="Adafactor"
            if ('OPTIMIZERS.register_module(name=\'Adafactor\'' in line or 
                'OPTIMIZERS.register_module(name="Adafactor"' in line):
                
                # 检查前几行是否有 try
                has_try = False
                for j in range(max(0, i - 5), i):
                    if 'try:' in lines[j]:
                        has_try = True
                        break
                
                if not has_try:
                    # 添加 try-except 包装
                    new_lines.append(f"{' ' * indent}# Fixed: PyTorch 2.9+ compatibility - skip if already registered\n")
                    new_lines.append(f"{' ' * indent}try:\n")
                    # 原行增加缩进
                    new_lines.append(f"{' ' * (indent + 4)}{stripped}")
                    i += 1
                    # 查找对应的 except 位置（通常是下一个非空行或函数结束）
                    # 简单处理：添加 except
                    if i < len(lines) and lines[i].strip() == '':
                        new_lines.append(lines[i])
                        i += 1
                    new_lines.append(f"{' ' * indent}except KeyError:\n")
                    new_lines.append(f"{' ' * (indent + 4)}# Adafactor already registered in torch.optim (PyTorch 2.9+), skip\n")
                    new_lines.append(f"{' ' * (indent + 4)}pass\n")
                    fixed = True
                    continue
        
        new_lines.append(line)
        i += 1
    
    # 如果修改了内容，写回文件
    if fixed:
        with open(builder_file, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        print("✓ 修复完成！")
        return True
    else:
        print("未找到需要修复的代码，可能代码结构不同")
        print("尝试使用更通用的修复方法...")
        return fix_adafactor_generic(builder_file, lines)

def fix_adafactor_generic(file_path, lines):
    """通用修复方法：直接替换注册行"""
    new_lines = []
    fixed = False
    
    for i, line in enumerate(lines):
        # 查找所有包含 Adafactor 注册的行
        if 'OPTIMIZERS.register_module' in line and 'Adafactor' in line:
            # 检查是否已经有 try-except
            has_try = False
            for j in range(max(0, i - 10), i):
                if 'try:' in lines[j] and 'Adafactor' in ''.join(lines[j:i+1]):
                    has_try = True
                    break
            
            if not has_try:
                indent = len(line) - len(line.lstrip())
                # 添加 try-except 包装
                new_lines.append(f"{' ' * indent}# Fixed: PyTorch 2.9+ compatibility\n")
                new_lines.append(f"{' ' * indent}try:\n")
                new_lines.append(f"{' ' * (indent + 4)}{line.lstrip()}")
                # 添加 except
                if i + 1 < len(lines) and lines[i + 1].strip() == '':
                    new_lines.append(lines[i + 1])
                    i += 1
                new_lines.append(f"{' ' * indent}except KeyError:\n")
                new_lines.append(f"{' ' * (indent + 4)}pass  # Adafactor already registered\n")
                fixed = True
                continue
        
        new_lines.append(line)
    
    if fixed:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        print("✓ 使用通用方法修复完成！")
        return True
    
    return False

def main():
    print("=" * 60)
    print("修复 mmengine Adafactor 重复注册问题")
    print("适用于: mmengine 0.10.1 + PyTorch 2.9+")
    print("=" * 60)
    print()
    
    try:
        success = fix_adafactor_registry()
        if success:
            print("\n" + "=" * 60)
            print("修复成功！现在可以正常使用 mmengine 了。")
            print("=" * 60)
            print("\n测试修复:")
            print("  python -c \"from mmengine.runner import Runner; print('成功!')\"")
            print("\n如果遇到问题，可以使用备份文件恢复:")
            print("  找到 mmengine 安装路径，然后:")
            print("  cp <mmengine_path>/optim/optimizer/builder.py.backup \\")
            print("     <mmengine_path>/optim/optimizer/builder.py")
        else:
            print("\n修复未完成，请检查错误信息")
            print("可能需要手动修改 mmengine 源代码")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()

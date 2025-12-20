#!/usr/bin/env python3
"""
快速修复 mmengine 0.10.1 与 PyTorch 2.9+ 的 Adafactor 重复注册问题

使用方法:
    激活 conda 环境后运行: python fix_mmengine.py
"""

import sys
import shutil
from pathlib import Path

def main():
    try:
        import mmengine
        mmengine_path = Path(mmengine.__file__).parent
    except ImportError:
        print("错误: 未找到 mmengine")
        print("请先激活 conda 环境并安装 mmengine==0.10.1")
        sys.exit(1)
    
    builder_file = mmengine_path / 'optim' / 'optimizer' / 'builder.py'
    
    if not builder_file.exists():
        print(f"错误: 未找到文件 {builder_file}")
        sys.exit(1)
    
    print(f"找到 mmengine: {mmengine_path}")
    print(f"修复文件: {builder_file}")
    
    # 备份
    backup_file = builder_file.with_suffix('.backup')
    if not backup_file.exists():
        shutil.copy2(builder_file, backup_file)
        print(f"✓ 已备份: {backup_file}")
    
    # 读取文件
    with open(builder_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查是否已修复
    if '# Fixed: PyTorch 2.9+' in content:
        print("文件已经修复过，跳过")
        return
    
    # 查找 Adafactor 注册行并修复
    lines = content.split('\n')
    new_lines = []
    fixed = False
    
    for i, line in enumerate(lines):
        # 查找包含 Adafactor 注册的行
        if ('OPTIMIZERS.register_module' in line and 
            'Adafactor' in line and 
            'try:' not in lines[max(0, i-5):i]):
            
            indent = len(line) - len(line.lstrip())
            
            # 添加 try-except
            new_lines.append(' ' * indent + 'try:')
            new_lines.append(' ' * (indent + 4) + '# Fixed: PyTorch 2.9+ compatibility')
            new_lines.append(' ' * (indent + 4) + line.lstrip())
            
            # 添加 except
            new_lines.append(' ' * indent + 'except KeyError:')
            new_lines.append(' ' * (indent + 4) + 'pass  # Adafactor already registered in torch.optim')
            
            fixed = True
        else:
            new_lines.append(line)
    
    if fixed:
        with open(builder_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(new_lines))
        print("✓ 修复完成！")
        
        # 测试
        print("\n测试修复...")
        try:
            from mmengine.runner import Runner
            print("✓ mmengine 导入成功！")
        except Exception as e:
            print(f"✗ 测试失败: {e}")
            print(f"可以恢复备份: cp {backup_file} {builder_file}")
    else:
        print("未找到需要修复的代码")

if __name__ == '__main__':
    main()


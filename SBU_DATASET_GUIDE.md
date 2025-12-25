# SBU数据集设置 - 标准路径

## 标准路径结构

SBU数据集应该放在：
```
/root/autodl-tmp/code/VMamba/data/sbu/
├── img/          # 输入图像 (.jpg)
│   ├── 0001.jpg
│   ├── 0002.jpg
│   └── ...
└── label/        # 标注图像 (.png)
    ├── 0001.png
    ├── 0002.png
    └── ...
```

---

## 快速设置（3种方案）

### 方案1: 数据集已在其他位置

如果您的SBU数据集在 `/path/to/SBU`，运行：

```bash
cd /root/autodl-tmp/code/VMamba

# 创建符号链接
mkdir -p data
ln -s /path/to/SBU data/sbu

# 验证
ls -la data/sbu/img/ | head
ls -la data/sbu/label/ | head
```

### 方案2: 数据集在autodl相关目录

```bash
cd /root/autodl-tmp/code/VMamba

# 尝试自动链接autodl-fs中的数据集
AUTODL_PATH="/root/autodl-fs/dataset/SBU"

if [ -d "$AUTODL_PATH" ]; then
    mkdir -p data
    ln -s "$AUTODL_PATH" data/sbu
    echo "✓ 数据集链接成功"
else
    echo "请提供实际数据集路径"
fi
```

### 方案3: 上传数据集到标准位置

```bash
# 创建目录
mkdir -p /root/autodl-tmp/code/VMamba/data/sbu/img
mkdir -p /root/autodl-tmp/code/VMamba/data/sbu/label

# 上传或复制数据集到这些目录
# 图像 → data/sbu/img/
# 标注 → data/sbu/label/
```

---

## 验证数据集

运行以下命令检查：

```bash
cd /root/autodl-tmp/code/VMamba

# 检查目录
ls -la data/sbu/

# 统计文件
echo "图像: $(ls data/sbu/img 2>/dev/null | wc -l)"
echo "标注: $(ls data/sbu/label 2>/dev/null | wc -l)"

# 应该看到:
# 图像: 4000+
# 标注: 4000+
```

---

## 开始训练

数据集准备好后：

```bash
cd /root/autodl-tmp/code/VMamba/segmentation

# 多GPU训练
bash tools/dist_train.sh configs/sbu/sbu_shadow_dual_stream_base_40k.py 4

# 或单GPU
python tools/train.py configs/sbu/sbu_shadow_dual_stream_base_40k.py
```

---

## 📞 快速命令

```bash
# 一键设置脚本（选择方案1）
cd /root/autodl-tmp/code/VMamba
bash setup_sbu_dataset.sh

# 或创建测试数据集（验证模型）
python3 create_test_dataset.py
```

---

按照标准路径 `data/sbu/img` 和 `data/sbu/label` 配置即可！

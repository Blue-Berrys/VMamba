# ✅ SBU数据集路径配置 - 实际路径

## 实际数据集结构

```
data/
└── SBU-shadow/
    ├── README
    ├── SBU-Test/                    # 测试集
    └── SBUTrain4KRecoveredSmall/      # 训练集 (4000张)
```

---

## 🚀 快速修复（2步）

### 步骤1: 创建符号链接

```bash
cd /root/autodl-tmp/code/VMamba

# 创建符号链接
ln -s data/SBU-shadow data/sbu

# 验证链接
ls -la data/sbu
```

应该看到：
```
data/sbu -> data/SBU-shadow
```

### 步骤2: 验证数据集结构

```bash
# 检查训练集
ls data/SBU-shadow/SBUTrain4KRecoveredSmall/ | head -20

# 检查是否有img和label子目录
# 或者直接在目录下
```

---

## 📝 可能的内部结构

### 结构A: 有子目录
```
SBUTrain4KRecoveredSmall/
├── img/           # 图像
└── label/         # 标注
```

### 结构B: 无子目录（图像和标注混合）
```
SBUTrain4KRecoveredSmall/
├── 0001.jpg
├── 0001.png
├── 0002.jpg
├── 0002.png
└── ...
```

---

## 🔍 检测实际结构

运行检测脚本：

```bash
cd /root/autodl-tmp/code/VMamba
bash fix_sbu_path.sh
```

这个脚本会：
1. 检查 `data/SBU-shadow/` 是否存在
2. 显示训练集和测试集目录
3. 检查内部结构
4. 自动创建符号链接
5. 验证配置

---

## 🎯 开始训练

符号链接创建后，直接开始训练：

```bash
cd /root/autodl-tmp/code/VMamba/segmentation

# 多GPU训练
bash tools/dist_train.sh configs/sbu/sbu_shadow_dual_stream_base_40k.py 4
```

---

## 📋 配置文件已更新

配置文件 `sbu_shadow_dual_stream_base_40k.py` 中的路径：
```python
dataset=dict(
    type='SBUDataset',
    data_root='data/sbu',          # → 实际指向 data/SBU-shadow
    data_prefix=dict(
        img_path='img',             # → data/SBU-shadow/.../img
        seg_map_path='label'       # → data/SBU-shadow/.../label
    )
)
```

---

## ✅ 一键设置

```bash
cd /root/autodl-tmp/code/VMamba

# 方法1: 自动检测并设置
bash fix_sbu_path.sh

# 方法2: 手动创建链接
ln -s data/SBU-shadow data/sbu

# 然后开始训练
cd segmentation
bash tools/dist_train.sh configs/sbu/sbu_shadow_dual_stream_base_40k.py 4
```

---

**总结**: 创建 `data/sbu` → `data/SBU-shadow` 的符号链接即可！

# ✅ 数据集缺失问题 - 快速解决方案

## 问题诊断

好消息！模型已经成功构建了！ 🎉

现在的问题是：
```
FileNotFoundError: [Errno 2] No such file or directory: 'data/sbu/img'
```

**原因**: SBU数据集还没有准备好

---

## 🚀 解决方案（3选1）

### 方案1: 创建测试数据集（推荐用于验证模型）

**目的**: 快速验证模型配置是否正确

```bash
cd /root/autodl-tmp/code/VMamba

# 创建临时测试数据集
python3 create_test_dataset.py

# 开始训练测试
cd segmentation
bash tools/dist_train.sh configs/sbu/sbu_shadow_dual_stream_base_40k.py 4
```

**预期效果**:
- 模型可以正常训练（虽然数据是随机的）
- 验证模型配置、数据流、训练流程都正确
- 训练几个iteration后停止，然后准备真实数据集

---

### 方案2: 下载真实SBU数据集

#### 下载方式1: 官方网站
```
网址: https://www.cs.sbu.edu/stuliu/iPhone-shadow-dataset/
```

#### 下载方式2: 百度网盘（如果有）
```
链接: https://pan.baidu.com/s/xxx
提取码: xxx
```

#### 数据集准备步骤：

```bash
cd /root/autodl-tmp/code/VMamba

# 1. 下载数据集到某个目录
DATASET_DIR="/root/autodl-tmp/datasets/SBU"
mkdir -p $DATASET_DIR

# 2. 解压数据集
# (假设下载的是 sbu_shadow.zip)
unzip sbu_shadow.zip -d $DATASET_DIR

# 3. 创建软链接
mkdir -p data
ln -s $DATASET_DIR data/sbu

# 4. 验证数据集
ls -la data/sbu/img/ | head
ls -la data/sbu/label/ | head

# 5. 开始训练
cd segmentation
bash tools/dist_train.sh configs/sbu/sbu_shadow_dual_stream_base_40k.py 4
```

---

### 方案3: 使用已有数据集

如果您已经有SBU数据集在其他位置：

```bash
cd /root/autodl-tmp/code/VMamba

# 修改配置文件中的数据路径
vim segmentation/configs/sbu/sbu_shadow_dual_stream_base_40k.py

# 找到这一行（约208-213行）:
# train_dataloader=dict(
#     dataset=dict(
#         data_root='data/sbu',  # ← 修改这里
#         ...

# 改为实际路径:
#         data_root='/your/actual/path/to/sbu',
#         ...

# 或者创建软链接
ln -s /your/actual/path/to/sbu data/sbu
```

---

## 🔍 验证数据集

运行以下命令检查数据集是否正确：

```bash
cd /root/autodl-tmp/code/VMamba

# 检查目录结构
ls -la data/sbu/
ls -la data/sbu/img/ | head -10
ls -la data/sbu/label/ | head -10

# 统计文件数量
echo "图像数量: $(ls data/sbu/img | wc -l)"
echo "标注数量: $(ls data/sbu/label | wc -l)"

# 应该看到：
# 图像数量: 4000+ (训练集)
# 标注数量: 4000+
```

---

## ✅ 数据集目录结构

正确的SBU数据集结构应该是：

```
data/sbu/
├── img/           # 输入图像 (.jpg)
│   ├── 0001.jpg
│   ├── 0002.jpg
│   └── ...
└── label/         # 标注图像 (.png)
    ├── 0001.png   # 0=非阴影(白色), 255=阴影(黑色)
    ├── 0002.png
    └── ...
```

---

## 🎯 推荐流程

### 第一步: 快速验证（5分钟）
```bash
cd /root/autodl-tmp/code/VMamba
python3 create_test_dataset.py
cd segmentation
bash tools/dist_train.sh configs/sbu/sbu_shadow_dual_stream_base_40k.py 1

# 看到训练日志后按Ctrl+C停止
# 这证明模型配置正确！
```

### 第二步: 准备真实数据集
- 下载SBU数据集
- 解压并放到正确位置
- 验证文件数量

### 第三步: 正式训练
```bash
cd /root/autodl-tmp/code/VMamba/segmentation
bash tools/dist_train.sh configs/sbu/sbu_shadow_dual_stream_base_40k.py 4
```

---

## 📊 训练监控

```bash
# 启动TensorBoard
tensorboard --logdir=work_dirs/sbu_shadow_detection_dual_stream_base

# 在浏览器访问
http://localhost:6006
```

---

## ⚠ 重要提醒

1. **测试数据集**: 仅用于验证模型，不会产生有意义的结果
2. **真实数据集**: 必须使用真实的SBU数据集进行正式训练
3. **数据路径**: 确保图像和标注文件名一一对应

---

## 🆘 需要帮助？

### 问题1: 不知道数据集在哪里
```bash
# 搜索整个文件系统
find / -type d -name "SBU*" 2>/dev/null
find / -type d -name "shadow*" 2>/dev/null
```

### 问题2: 数据集格式不对
```python
# 检查图像格式
from PIL import Image
import os

img_dir = "data/sbu/img"
label_dir = "data/sbu/label"

for f in os.listdir(img_dir)[:10]:
    img = Image.open(os.path.join(img_dir, f))
    print(f"{f}: {img.size}, {img.mode}")

# 应该看到:
# 0001.jpg: (640, 480), RGB
# ...
```

### 问题3: 文件名不匹配
```bash
# 检查文件名是否对应
ls data/sbu/img/ | sort > /tmp/imgs.txt
ls data/sbu/label/ | sort > /tmp/labels.txt
diff /tmp/imgs.txt /tmp/labels.txt

# 应该只有扩展名不同（.jpg vs .png）
```

---

## 🎉 总结

**当前状态**: ✅ 模型配置正确，构建成功！

**下一步**: 准备数据集

**快速验证**:
```bash
python3 create_test_dataset.py  # 2分钟
# 开始测试训练，验证一切正常
```

**正式训练**: 准备真实SBU数据集后开始

---

**准备好数据集后，就可以开始正式训练了！** 🚀

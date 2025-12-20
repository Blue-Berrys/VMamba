# PyTorch 快速安装指南（解决下载慢的问题）

## 问题
PyTorch 官方源下载速度很慢（~79 kB/s），需要 3+ 小时。

## 解决方案

### 方法 1: 使用 conda 安装（推荐，最快）

conda 的镜像源通常比 pip 更稳定，而且 conda 会自动处理依赖：

```bash
conda activate vmamba

# 使用 conda-forge 镜像（清华镜像）
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch

# 安装 PyTorch
conda install pytorch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 pytorch-cuda=12.8 -c pytorch -c nvidia -y
```

### 方法 2: 使用 pip + 镜像源（脚本已支持）

运行脚本时选择使用镜像源：

```bash
bash install_cuda128_pytorch29.sh
# 当询问是否使用镜像源时，输入 y
```

### 方法 3: 手动下载 wheel 文件（最稳定）

如果网络实在太慢，可以手动下载 wheel 文件：

```bash
# 1. 下载 wheel 文件（使用浏览器或 wget，支持断点续传）
# PyTorch 2.9.1 CUDA 12.8 下载链接：
# https://download.pytorch.org/whl/cu128/torch-2.9.1%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl
# https://download.pytorch.org/whl/cu128/torchvision-0.24.1%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl
# https://download.pytorch.org/whl/cu128/torchaudio-2.9.1%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl

# 2. 使用 pip 安装本地文件
pip install torch-2.9.1+cu128-cp310-cp310-manylinux_2_28_x86_64.whl
pip install torchvision-0.24.1+cu128-cp310-cp310-manylinux_2_28_x86_64.whl
pip install torchaudio-2.9.1+cu128-cp310-cp310-manylinux_2_28_x86_64.whl
```

### 方法 4: 使用代理（如果有）

```bash
# 设置代理
export http_proxy=http://your-proxy:port
export https_proxy=http://your-proxy:port

# 然后运行安装
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
```

### 方法 5: 使用 aria2 多线程下载（推荐）

```bash
# 安装 aria2
sudo apt install aria2  # Ubuntu/Debian
# 或
brew install aria2     # macOS

# 使用 aria2 下载（支持多线程和断点续传）
aria2c -x 16 -s 16 \
  https://download.pytorch.org/whl/cu128/torch-2.9.1%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl

# 然后安装
pip install torch-2.9.1+cu128-cp310-cp310-manylinux_2_28_x86_64.whl
```

## 推荐方案

**对于国内用户，强烈推荐使用 conda 安装**（方法 1），因为：
1. conda 镜像源更稳定
2. 自动处理 CUDA 依赖
3. 下载速度通常更快

## 验证安装

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```


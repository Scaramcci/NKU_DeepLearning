#!/bin/zsh

# 创建必要的目录
mkdir -p models images

# 训练模型
echo "开始训练GAN模型..."
python train.py

# 分析潜在空间
echo "\n开始分析潜在空间..."
python analyze.py

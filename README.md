# Transformer 机器翻译模型

一个基于 PyTorch 实现的 Transformer 模型，用于英语到德语的机器翻译任务，使用 IWSLT2017 数据集进行训练。

## 模型架构

本项目实现了标准的 Transformer 架构，包含以下核心组件：

- **多头自注意力机制**：使用缩放点积注意力，支持多头并行计算
- **位置编码**：正弦余弦位置编码，为序列添加位置信息
- **编码器-解码器结构**：6层编码器和6层解码器
- **残差连接和层归一化**：每个子层后都有残差连接和层归一化
- **位置前馈网络**：每个注意力层后接两层前馈网络


## 环境要求

```bash
conda create -n transformer python=3.10
conda activate transformer
pip install torch==2.0.0 torchvision torchaudio
pip install transformers datasets tqdm tensorboard
```

## 数据集

使用 IWSLT2017 英语-德语翻译数据集。数据集应该按以下结构组织：

```
data/
├── train/
│   ├── *.arrow
│   └── *.json
├── validation/
│   ├── *.arrow
│   └── *.json
└── test/
    ├── *.arrow
    └── *.json
```

## 快速开始

### 1. 训练模型

```bash
python train.py \
  --data_dir /path/to/your/data \
  --batch_size 64 \
  --epochs 10 \
  --lr 0.0001 \
  --d_model 512 \
  --num_heads 8 \
  --num_layers 6 \
```

### 2. 进行消融实验（无位置编码）

```bash
python train.py \
  --data_dir /path/to/your/data \
  --no_pos_encoding \
  --batch_size 64 \
  --epochs 10 \
  --lr 0.0001 \
```

### 3. 使用不同模型维度

```bash
python train.py \
  --data_dir /path/to/your/data \
  --d_model 256 \
  --batch_size 64 \
  --epochs 10 \
  --lr 0.0001 \
```

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--data_dir` | str | 必填 | 数据集路径 |
| `--batch_size` | int | 32 | 批次大小 |
| `--epochs` | int | 10 | 训练轮数 |
| `--lr` | float | 0.0001 | 学习率 |
| `--d_model` | int | 512 | 模型维度 |
| `--num_heads` | int | 8 | 注意力头数 |
| `--num_layers` | int | 6 | 编码器/解码器层数 |
| `--d_ff` | int | 2048 | 前馈网络维度 |
| `--max_len` | int | 100 | 最大序列长度 |
| `--dropout` | float | 0.1 | Dropout概率 |
| `--max_samples` | int | 50000 | 最大训练样本数 |
| `--seed` | int | 42 | 随机种子 |
| `--no_pos_encoding` | flag | False | 禁用位置编码（消融实验） |

## 输出文件

训练完成后会生成以下文件：

- `best_transformer_model.pth` - 最佳模型权重
- `training_curves.png` - 训练曲线图
- `training_metrics.csv` - 训练指标数据
- 控制台输出的翻译示例

## 评估指标

模型在以下指标上进行评估：

1. **交叉熵损失** - 衡量预测分布与真实分布的差异
2. **困惑度** - `exp(loss)`，表示模型的不确定性
3. **准确率** - 下一个词预测的正确率
4. **BLEU分数** - 衡量翻译质量的常用指标

## 实验结果

在 IWSLT2017 英语-德语测试集上的典型结果：

| 模型配置 | 测试损失 | 困惑度 | BLEU分数 |
|----------|----------|--------|-----------|
| 标准Transformer |3.58 | 37.60 | 17.06 |
| 无位置编码 | 3.88| 49.88 | 7.57|

## 引用

如果您使用了本项目，请引用原始 Transformer 论文：

```bibtex
@inproceedings{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia},
  booktitle={Advances in neural information processing systems},
  pages={5998--6008},
  year={2017}
}
```


## 致谢

- 感谢 [PyTorch](https://pytorch.org/) 团队提供的深度学习框架
- 感谢 [Hugging Face](https://huggingface.co/) 提供的 datasets 库
- 感谢 IWSLT 2017 数据集提供者

---

*如有问题，请提交 Issue 或联系项目维护者。*

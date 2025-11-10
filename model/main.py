import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import torch
from torch import nn
import torch.nn.functional as F
import random
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import sacrebleu
import argparse
import os
import json
import pyarrow as pa
from datasets import load_dataset
import glob

# ---------------------------
# Transformer 组件实现（您的代码保持不变）
# ---------------------------

class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
'''
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
'''
    self.register_buffer('pe', torch.zeros(1, max_len, d_model))
    def forward(self, x):
        #x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, query, key, value, mask=None):
        batch_size, q_seq_len = query.size(0), query.size(1)
        k_seq_len = key.size(1)

        Q = self.w_q(query).view(batch_size, q_seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, k_seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, k_seq_len, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, q_seq_len, self.d_model)

        output = self.w_o(context)
        return output, attention_weights

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.net(x)

class AddNorm(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(AddNorm, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer_output):
        return self.norm(x + self.dropout(sublayer_output))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.add_norm1 = AddNorm(d_model, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.add_norm2 = AddNorm(d_model, dropout)

    def forward(self, x, mask=None):
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.add_norm1(x, attn_output)
        ffn_output = self.ffn(x)
        x = self.add_norm2(x, ffn_output)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.add_norm1 = AddNorm(d_model, dropout)
        self.enc_dec_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.add_norm2 = AddNorm(d_model, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.add_norm3 = AddNorm(d_model, dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.add_norm1(x, attn_output)
        attn_output, _ = self.enc_dec_attn(x, encoder_output, encoder_output, src_mask)
        x = self.add_norm2(x, attn_output)
        ffn_output = self.ffn(x)
        x = self.add_norm3(x, ffn_output)
        return x

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len=5000, dropout=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.embedding = Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        x = self.embedding(src)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len=5000, dropout=0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.embedding = Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.output_linear = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        x = self.embedding(tgt)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        x = self.output_linear(x)
        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, d_ff=2048,
                 num_encoder_layers=6, num_decoder_layers=6, max_len=5000, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, d_model, num_heads, d_ff, num_encoder_layers, max_len, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_heads, d_ff, num_decoder_layers, max_len, dropout)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        encoder_output = self.encoder(src, src_mask)
        decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        return decoder_output

# ---------------------------
# 数据相关工具
# ---------------------------

# 特殊 token 定义
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

def white_tokenize(text):
    return text.strip().lower().split()

def build_vocab_from_iterator(iterator, tokenizer, max_size=10000, min_freq=2):
    counter = Counter()
    for sent in iterator:
        tokens = tokenizer(sent)
        counter.update(tokens)
    most_common = [tok for tok, freq in counter.most_common(max_size) if freq >= min_freq]
    itos = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN] + most_common
    stoi = {tok: idx for idx, tok in enumerate(itos)}
    return stoi, itos

def load_iwslt_data(data_dir, split='train', max_samples=None):
    """加载IWSLT数据集并返回(src, tgt)对列表"""
    split_dir = os.path.join(data_dir, split)
    pairs = []
    
    # 查找arrow文件
    arrow_files = glob.glob(os.path.join(split_dir, "*.arrow"))
    if not arrow_files:
        raise FileNotFoundError(f"在 {split_dir} 中找不到.arrow文件")
    
    # 使用第一个找到的arrow文件
    arrow_file = arrow_files[0]
    print(f"加载 {split} 数据集: {arrow_file}")
    
    try:
        # 加载数据集
        dataset = load_dataset('arrow', data_files=arrow_file)['train']
        
        for i, item in enumerate(dataset):
            if max_samples and i >= max_samples:
                break
                
            # 根据不同的数据集格式处理字段名
            if 'translation' in item:
                src_text = item['translation']['en']
                tgt_text = item['translation']['de']
            elif 'en' in item and 'de' in item:
                src_text = item['en']
                tgt_text = item['de']
            else:
                # 尝试找到包含文本的字段
                text_fields = [k for k, v in item.items() if isinstance(v, str)]
                if len(text_fields) >= 2:
                    src_text = item[text_fields[0]]
                    tgt_text = item[text_fields[1]]
                else:
                    continue
            
            pairs.append((src_text, tgt_text))
            
    except Exception as e:
        print(f"加载数据集失败: {e}")
        return []
    
    return pairs

class IWSLTDataset(Dataset):
    def __init__(self, pairs, src_stoi, tgt_stoi, src_tok, tgt_tok, max_len=100):
        self.pairs = pairs
        self.src_stoi = src_stoi
        self.tgt_stoi = tgt_stoi
        self.src_tok = src_tok
        self.tgt_tok = tgt_tok
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def tokenize_to_ids(self, text, stoi, tokenizer, add_bos_eos=False):
        toks = tokenizer(text)
        ids = [stoi.get(t, stoi[UNK_TOKEN]) for t in toks]
        if add_bos_eos:
            ids = [stoi[BOS_TOKEN]] + ids + [stoi[EOS_TOKEN]]
        if len(ids) > self.max_len:
            ids = ids[:self.max_len]
            if add_bos_eos and ids[-1] != stoi[EOS_TOKEN]:
                ids[-1] = stoi[EOS_TOKEN]
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        src_ids = self.tokenize_to_ids(src, self.src_stoi, self.src_tok, add_bos_eos=False)
        tgt_ids = self.tokenize_to_ids(tgt, self.tgt_stoi, self.tgt_tok, add_bos_eos=True)
        return src_ids, tgt_ids

def collate_fn(batch, pad_idx):
    srcs, tgts = zip(*batch)
    srcs_padded = torch.nn.utils.rnn.pad_sequence(srcs, batch_first=True, padding_value=pad_idx)
    tgts_padded = torch.nn.utils.rnn.pad_sequence(tgts, batch_first=True, padding_value=pad_idx)
    return srcs_padded, tgts_padded

# ---------------------------
# Mask 构造函数
# ---------------------------
def create_padding_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)

def create_tgt_mask(tgt_seq, pad_idx):
    B, T = tgt_seq.size()
    padding_mask = (tgt_seq != pad_idx).unsqueeze(1).unsqueeze(2)
    causal = torch.tril(torch.ones((T, T), dtype=torch.bool, device=tgt_seq.device)).unsqueeze(0).unsqueeze(1)
    return padding_mask & causal



def train_epoch(model, dataloader, optimizer, criterion, device, pad_idx):
    model.train()
    total_loss = 0.0
    total_perplexity = 0.0
    total_accuracy = 0.0
    num_batches = 0
    
    for src, tgt in tqdm(dataloader, desc="Train", leave=False):
        src = src.to(device)
        tgt = tgt.to(device)
        src_mask = create_padding_mask(src, pad_idx).to(device)
        tgt_input = tgt[:, :-1].to(device)
        tgt_out = tgt[:, 1:].to(device)
        tgt_mask = create_tgt_mask(tgt_input, pad_idx).to(device)

        logits = model(src, tgt_input, src_mask, tgt_mask)
        logits_flat = logits.reshape(-1, logits.size(-1))
        tgt_out_flat = tgt_out.reshape(-1)

        loss = criterion(logits_flat, tgt_out_flat)
        
        # 计算困惑度和准确率
        perplexity = torch.exp(loss)
        predictions = logits_flat.argmax(dim=-1)
        non_pad_mask = tgt_out_flat != pad_idx
        correct = (predictions == tgt_out_flat) & non_pad_mask
        accuracy = correct.sum().float() / non_pad_mask.sum().float()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_perplexity += perplexity.item()
        total_accuracy += accuracy.item()
        num_batches += 1
        
    return (
        total_loss / num_batches,
        total_perplexity / num_batches,
        total_accuracy / num_batches
    )

def evaluate(model, dataloader, criterion, device, pad_idx):
    model.eval()
    total_loss = 0.0
    total_perplexity = 0.0
    total_accuracy = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for src, tgt in tqdm(dataloader, desc="Eval", leave=False):
            src = src.to(device)
            tgt = tgt.to(device)
            src_mask = create_padding_mask(src, pad_idx).to(device)
            tgt_input = tgt[:, :-1].to(device)
            tgt_out = tgt[:, 1:].to(device)
            tgt_mask = create_tgt_mask(tgt_input, pad_idx).to(device)

            logits = model(src, tgt_input, src_mask, tgt_mask)
            logits_flat = logits.reshape(-1, logits.size(-1))
            tgt_out_flat = tgt_out.reshape(-1)
            
            loss = criterion(logits_flat, tgt_out_flat)
            
            # 计算困惑度和准确率
            perplexity = torch.exp(loss)
            predictions = logits_flat.argmax(dim=-1)
            non_pad_mask = tgt_out_flat != pad_idx
            correct = (predictions == tgt_out_flat) & non_pad_mask
            accuracy = correct.sum().float() / non_pad_mask.sum().float()
            
            total_loss += loss.item()
            total_perplexity += perplexity.item()
            total_accuracy += accuracy.item()
            num_batches += 1
            
    return (
        total_loss / num_batches,
        total_perplexity / num_batches,
        total_accuracy / num_batches
    )

def greedy_decode(model, src, src_stoi, tgt_itos, pad_idx, bos_idx, eos_idx, device, max_len=50):
    model.eval()
    src = src.to(device)
    src_mask = create_padding_mask(src, pad_idx).to(device)
    memory = model.encoder(src, src_mask)

    ys = torch.tensor([[bos_idx]], dtype=torch.long, device=device)
    for i in range(max_len):
        tgt_mask = create_tgt_mask(ys, pad_idx).to(device)
        out = model.decoder(ys, memory, src_mask, tgt_mask)
        prob = out[:, -1, :]
        next_word = prob.argmax(dim=-1).unsqueeze(0)
        ys = torch.cat([ys, next_word], dim=1)
        if next_word.item() == eos_idx:
            break
    ids = ys.squeeze(0).tolist()
    if ids and ids[0] == bos_idx:
        ids = ids[1:]
    if eos_idx in ids:
        ids = ids[:ids.index(eos_idx)]
    tokens = [tgt_itos[i] if i < len(tgt_itos) else "<unk>" for i in ids]
    return " ".join(tokens)

def calculate_bleu(model, dataloader, src_stoi, tgt_itos, tgt_stoi, device, max_samples=100, max_len=50):
    """计算BLEU分数"""
    model.eval()
    hypotheses = []
    references = []
    
    pad_idx = src_stoi[PAD_TOKEN]
    bos_idx = tgt_stoi[BOS_TOKEN]
    eos_idx = tgt_stoi[EOS_TOKEN]
    
    with torch.no_grad():
        for i, (src, tgt) in enumerate(dataloader):
            if i >= max_samples:
                break
                
            for j in range(src.size(0)):
                src_single = src[j:j+1]
                tgt_single = tgt[j:j+1]
                
                # 生成翻译
                translation = greedy_decode(
                    model, src_single, src_stoi, tgt_itos, 
                    pad_idx, bos_idx, eos_idx, device, max_len
                )
                
                # 获取参考翻译
                tgt_tokens = []
                for idx in tgt_single[0].tolist():
                    if idx == eos_idx:
                        break
                    if idx != bos_idx and idx != pad_idx:
                        token = tgt_itos[idx] if idx < len(tgt_itos) else UNK_TOKEN
                        tgt_tokens.append(token)
                reference = " ".join(tgt_tokens)
                
                hypotheses.append(translation)
                references.append(reference)
    
    if len(hypotheses) > 0 and len(references) > 0:
        try:
            bleu = sacrebleu.corpus_bleu(hypotheses, [references])
            return {
                'sacrebleu_score': bleu.score,
                'num_samples': len(hypotheses)
            }
        except:
            return {
                'sacrebleu_score': 0,
                'num_samples': len(hypotheses)
            }
    else:
        return {
            'sacrebleu_score': 0,
            'num_samples': 0
        }

def plot_training_curves(train_losses, val_losses, train_perplexities, val_perplexities,
                        train_accuracies, val_accuracies, val_bleu_scores=None, 
                        save_path='training_curves.png'):
    """绘制完整的训练曲线"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    epochs = range(1, len(train_losses) + 1)
    
    # 损失曲线
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss', fontsize=14)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 困惑度曲线
    ax2.plot(epochs, train_perplexities, 'b-', label='Training Perplexity', linewidth=2)
    ax2.plot(epochs, val_perplexities, 'r-', label='Validation Perplexity', linewidth=2)
    ax2.set_title('Training and Validation Perplexity', fontsize=14)
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Perplexity')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # 准确率曲线
    ax3.plot(epochs, train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
    ax3.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
    ax3.set_title('Training and Validation Accuracy', fontsize=14)
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Accuracy')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # BLEU分数曲线
    if val_bleu_scores:
        ax4.plot(epochs, val_bleu_scores, 'g-', label='Validation BLEU', linewidth=2)
        ax4.set_title('Validation BLEU Score', fontsize=14)
        ax4.set_xlabel('Epochs')
        ax4.set_ylabel('BLEU Score')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"训练曲线已保存到: {save_path}")

def save_metrics_to_csv(train_losses, val_losses, train_perplexities, val_perplexities,
                       train_accuracies, val_accuracies, val_bleu_scores=None, 
                       save_path='training_metrics.csv'):
    """保存所有指标到CSV文件"""
    with open(save_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 写入表头
        if val_bleu_scores:
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'train_perplexity', 
                            'val_perplexity', 'train_accuracy', 'val_accuracy', 'val_bleu'])
            # 写入数据
            for epoch, (tl, vl, tp, vp, ta, va, vb) in enumerate(
                zip(train_losses, val_losses, train_perplexities, val_perplexities,
                    train_accuracies, val_accuracies, val_bleu_scores), 1):
                writer.writerow([epoch, tl, vl, tp, vp, ta, va, vb])
        else:
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'train_perplexity', 
                            'val_perplexity', 'train_accuracy', 'val_accuracy'])
            # 写入数据
            for epoch, (tl, vl, tp, vp, ta, va) in enumerate(
                zip(train_losses, val_losses, train_perplexities, val_perplexities,
                    train_accuracies, val_accuracies), 1):
                writer.writerow([epoch, tl, vl, tp, vp, ta, va])
    
    print(f"训练指标已保存到: {save_path}")

def print_translation_examples(model, dataloader, src_stoi, tgt_itos, tgt_stoi, device, num_examples=3):
    """打印翻译示例"""
    model.eval()
    examples_printed = 0
    
    pad_idx = src_stoi[PAD_TOKEN]
    bos_idx = tgt_stoi[BOS_TOKEN]
    eos_idx = tgt_stoi[EOS_TOKEN]
    
    print("\n" + "="*60)
    print("翻译示例")
    print("="*60)
    
    with torch.no_grad():
        for src, tgt in dataloader:
            for j in range(src.size(0)):
                if examples_printed >= num_examples:
                    return
                    
                src_single = src[j:j+1]
                tgt_single = tgt[j:j+1]
                
                # 获取源文本
                src_tokens = []
                for idx in src_single[0].tolist():
                    if idx != pad_idx:
                        token = src_itos[idx] if idx < len(src_itos) else UNK_TOKEN
                        src_tokens.append(token)
                source_text = " ".join(src_tokens)
                
                # 获取参考翻译
                tgt_tokens = []
                for idx in tgt_single[0].tolist():
                    if idx == eos_idx:
                        break
                    if idx != bos_idx and idx != pad_idx:
                        token = tgt_itos[idx] if idx < len(tgt_itos) else UNK_TOKEN
                        tgt_tokens.append(token)
                reference_text = " ".join(tgt_tokens)
                
                # 生成模型翻译
                translation = greedy_decode(
                    model, src_single, src_stoi, tgt_itos, 
                    pad_idx, bos_idx, eos_idx, device
                )
                
                print(f"源文本: {source_text}")
                print(f"参考翻译: {reference_text}")
                print(f"模型翻译: {translation}")
                print("-" * 60)
                
                examples_printed += 1

# 在main函数中修改训练循环部分
def main():
    # 配置参数
    parser = argparse.ArgumentParser(description='Transformer Training')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to IWSLT2017 dataset directory')
    parser.add_argument('--max_samples', type=int, default=50000, help='Maximum number of samples to use for training')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--max_len', type=int, default=100)
    parser.add_argument('--dropout', type=float, default=0.3)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载数据
    print("加载训练数据...")
    train_pairs = load_iwslt_data(args.data_dir, 'train', args.max_samples)
    val_pairs = load_iwslt_data(args.data_dir, 'validation', args.max_samples // 10)
    test_pairs = load_iwslt_data(args.data_dir, 'test', args.max_samples // 10)
    
    print(f"训练样本数: {len(train_pairs)}")
    print(f"验证样本数: {len(val_pairs)}")
    print(f"测试样本数: {len(test_pairs)}")
    
    # 构建词表
    print("构建词表...")
    src_texts = [pair[0] for pair in train_pairs]
    tgt_texts = [pair[1] for pair in train_pairs]
    
    src_stoi, src_itos = build_vocab_from_iterator(src_texts, white_tokenize, max_size=10000, min_freq=2)
    tgt_stoi, tgt_itos = build_vocab_from_iterator(tgt_texts, white_tokenize, max_size=10000, min_freq=2)
    
    src_vocab_size = len(src_itos)
    tgt_vocab_size = len(tgt_itos)
    
    print(f"源语言词表大小: {src_vocab_size}")
    print(f"目标语言词表大小: {tgt_vocab_size}")
    
    # 创建数据集和数据加载器
    train_dataset = IWSLTDataset(train_pairs, src_stoi, tgt_stoi, white_tokenize, white_tokenize, args.max_len)
    val_dataset = IWSLTDataset(val_pairs, src_stoi, tgt_stoi, white_tokenize, white_tokenize, args.max_len)
    test_dataset = IWSLTDataset(test_pairs, src_stoi, tgt_stoi, white_tokenize, white_tokenize, args.max_len)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                             collate_fn=lambda x: collate_fn(x, pad_idx=src_stoi[PAD_TOKEN]))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                           collate_fn=lambda x: collate_fn(x, pad_idx=src_stoi[PAD_TOKEN]))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=lambda x: collate_fn(x, pad_idx=src_stoi[PAD_TOKEN]))
    
    # 初始化模型
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        num_encoder_layers=args.num_layers,
        num_decoder_layers=args.num_layers,
        max_len=args.max_len,
        dropout=args.dropout
    ).to(device)
    
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 定义优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=src_stoi[PAD_TOKEN])
    
    # 初始化记录列表
    train_losses = []
    val_losses = []
    train_perplexities = []
    val_perplexities = []
    train_accuracies = []
    val_accuracies = []
    val_bleu_scores = []
    
    # 训练循环
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        train_loss, train_perplexity, train_accuracy = train_epoch(
            model, train_loader, optimizer, criterion, device, src_stoi[PAD_TOKEN]
        )
        val_loss, val_perplexity, val_accuracy = evaluate(
            model, val_loader, criterion, device, src_stoi[PAD_TOKEN]
        )
        
        # 记录指标
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_perplexities.append(train_perplexity)
        val_perplexities.append(val_perplexity)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        
        print(f"训练 - 损失: {train_loss:.4f}, 困惑度: {train_perplexity:.2f}, 准确率: {train_accuracy:.4f}")
        print(f"验证 - 损失: {val_loss:.4f}, 困惑度: {val_perplexity:.2f}, 准确率: {val_accuracy:.4f}")
        
        # 每隔几个epoch计算一次BLEU分数
        if epoch % 2 == 0 or epoch == args.epochs - 1:
            print("计算验证集BLEU分数...")
            bleu_results = calculate_bleu(
                model, val_loader, src_stoi, tgt_itos, tgt_stoi, 
                device, max_samples=100
            )
            val_bleu_scores.append(bleu_results['sacrebleu_score'])
            print(f"BLEU分数: {bleu_results['sacrebleu_score']:.2f}")
        else:
            val_bleu_scores.append(val_bleu_scores[-1] if val_bleu_scores else 0)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'src_stoi': src_stoi,
                'tgt_itos': tgt_itos,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_perplexity': train_perplexity,
                'val_perplexity': val_perplexity,
                'val_bleu': val_bleu_scores[-1],
                'args': args
            }, 'best_transformer_model.pth')
            print("保存最佳模型!")
    
    # 测试
    print("\n测试模型...")
    test_loss, test_perplexity, test_accuracy = evaluate(
        model, test_loader, criterion, device, src_stoi[PAD_TOKEN]
    )
    
    # 计算测试集BLEU分数
    test_bleu = calculate_bleu(
        model, test_loader, src_stoi, tgt_itos, tgt_stoi, 
        device, max_samples=200
    )
    
    print(f"测试损失: {test_loss:.4f}")
    print(f"测试困惑度: {test_perplexity:.2f}")
    print(f"测试准确率: {test_accuracy:.4f}")
    print(f"测试BLEU分数: {test_bleu['sacrebleu_score']:.2f}")
    
    # 绘制训练曲线
    print("\n绘制训练曲线...")
    plot_training_curves(
        train_losses, val_losses, train_perplexities, val_perplexities,
        train_accuracies, val_accuracies, val_bleu_scores
    )
    save_metrics_to_csv(
        train_losses, val_losses, train_perplexities, val_perplexities,
        train_accuracies, val_accuracies, val_bleu_scores
    )
    
    # 打印训练总结
    print("\n=== 训练总结 ===")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"最终测试损失: {test_loss:.4f}")
    print(f"最终测试困惑度: {test_perplexity:.2f}")
    print(f"最终测试准确率: {test_accuracy:.4f}")
    print(f"最终测试BLEU: {test_bleu['sacrebleu_score']:.2f}")
    print(f"训练轮次: {args.epochs}")
    
    # 打印翻译示例
    print_translation_examples(
        model, test_loader, src_stoi, tgt_itos, tgt_stoi, 
        device, num_examples=3
    )

if __name__ == "__main__":
    main()
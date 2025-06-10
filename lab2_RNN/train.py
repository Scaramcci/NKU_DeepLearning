import torch
import torch.nn as nn
import time
import math
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from utils.data_loader import load_data, randomTrainingExample, lineToTensor
from models.rnn import RNN
from models.lstm import LSTM

# 创建保存图表的目录
if not os.path.exists('results'):
    os.makedirs('results')

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def categoryFromOutput(output, all_categories):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

def train_model(model, category_tensor, line_tensor, criterion, learning_rate=0.005):
    if isinstance(model, LSTM):
        hidden = model.initHidden()  # 返回(h, c)
        output = None
        
        for i in range(line_tensor.size(0)):
            output, hidden = model(line_tensor[i], hidden)
            
    else:  # RNN
        hidden = model.initHidden()  # 只返回h
        
        for i in range(line_tensor.size(0)):
            output, hidden = model(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in model.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()

def train(model_type='rnn', n_iters=100000, print_every=5000, plot_every=1000, learning_rate=0.005):
    # 加载数据
    category_lines, all_categories, n_categories = load_data('data/names/*.txt')
    
    # 创建模型
    n_hidden = 128
    n_letters = 57  # len(all_letters)
    
    if model_type.lower() == 'lstm':
        model = LSTM(n_letters, n_hidden, n_categories)
    else:
        model = RNN(n_letters, n_hidden, n_categories)
    
    criterion = nn.NLLLoss()
    
    current_loss = 0
    all_losses = []
    accuracies = []
    start = time.time()
    
    for iter in range(1, n_iters + 1):
        category, line, category_tensor, line_tensor = randomTrainingExample(category_lines, all_categories)
        
        model.zero_grad()
        output, loss = train_model(model, category_tensor, line_tensor, criterion, learning_rate)
        current_loss += loss

        # 打印训练信息
        if iter % print_every == 0:
            guess, guess_i = categoryFromOutput(output, all_categories)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))
            
            # 计算当前准确率
            acc, _ = calculate_accuracy(model, category_lines, all_categories, n_samples=1000)
            accuracies.append(acc)

        # 记录损失
        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0
    
    # 保存损失曲线
    plt.figure()
    plt.plot(all_losses)
    plt.title(f'Training Loss ({model_type.upper()})')
    plt.xlabel('Iterations (x{})'.format(plot_every))
    plt.ylabel('Loss')
    plt.savefig(f'results/loss_{model_type.lower()}.png')
    plt.close()
    
    # 保存准确率曲线
    plt.figure()
    plt.plot(accuracies)
    plt.title(f'Training Accuracy ({model_type.upper()})')
    plt.xlabel('Iterations (x{})'.format(print_every))
    plt.ylabel('Accuracy')
    plt.savefig(f'results/accuracy_{model_type.lower()}.png')
    plt.close()
    
    # 计算并保存混淆矩阵
    confusion = calculate_confusion_matrix(model, category_lines, all_categories, n_categories)
    plot_confusion_matrix(confusion, all_categories, 
                         f'Confusion Matrix ({model_type.upper()})',
                         f'results/confusion_matrix_{model_type.lower()}.png')
    
    return model, all_losses, accuracies

def evaluate(model, line_tensor):
    with torch.no_grad():
        if isinstance(model, LSTM):
            hidden = model.initHidden()
            for i in range(line_tensor.size(0)):
                output, hidden = model(line_tensor[i], hidden)
        else:
            hidden = model.initHidden()
            for i in range(line_tensor.size(0)):
                output, hidden = model(line_tensor[i], hidden)
        return output

def calculate_confusion_matrix(model, category_lines, all_categories, n_categories):
    confusion = torch.zeros(n_categories, n_categories)
    n_confusion = 10000
    
    for i in range(n_confusion):
        category, line, category_tensor, line_tensor = randomTrainingExample(category_lines, all_categories)
        output = evaluate(model, line_tensor)
        guess, guess_i = categoryFromOutput(output, all_categories)
        category_i = all_categories.index(category)
        confusion[category_i][guess_i] += 1
    
    # 归一化
    for i in range(n_categories):
        confusion[i] = confusion[i] / confusion[i].sum()
    
    return confusion

def plot_confusion_matrix(confusion, all_categories, title, save_path):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)
    
    # 设置坐标轴
    ax.set_xticklabels([''] + all_categories, rotation=90)
    ax.set_yticklabels([''] + all_categories)
    
    # 强制在每个刻度处显示标签
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def calculate_accuracy(model, category_lines, all_categories, n_samples=1000):
    correct = 0
    accuracies = []
    
    for i in range(n_samples):
        category, line, category_tensor, line_tensor = randomTrainingExample(category_lines, all_categories)
        output = evaluate(model, line_tensor)
        guess, _ = categoryFromOutput(output, all_categories)
        
        if guess == category:
            correct += 1
        
        # 每100个样本记录一次准确率
        if (i + 1) % 100 == 0:
            accuracies.append(correct / (i + 1))
    
    return correct / n_samples, accuracies

if __name__ == '__main__':
    # 训练RNN
    print("Training RNN...")
    rnn_model, rnn_losses, rnn_accuracies = train(model_type='rnn')
    
    # 训练LSTM
    print("\nTraining LSTM...")
    lstm_model, lstm_losses, lstm_accuracies = train(model_type='lstm')
    
    # 保存损失对比曲线
    plt.figure(figsize=(10, 5))
    plt.plot(rnn_losses, label='RNN')
    plt.plot(lstm_losses, label='LSTM')
    plt.title('Training Loss Comparison')
    plt.xlabel('Iterations (x1000)')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('results/loss_comparison.png')
    plt.close()
    
    # 保存准确率对比曲线
    plt.figure(figsize=(10, 5))
    plt.plot(rnn_accuracies, label='RNN')
    plt.plot(lstm_accuracies, label='LSTM')
    plt.title('Training Accuracy Comparison')
    plt.xlabel('Iterations (x5000)')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('results/accuracy_comparison.png')
    plt.close()
    
    # 计算并绘制混淆矩阵
    confusion = calculate_confusion_matrix(lstm_model, category_lines, all_categories, n_categories)
    plot_confusion_matrix(confusion, all_categories, 'Confusion Matrix (LSTM)', 'results/confusion_matrix_lstm.png')
    
    # 计算准确率
    accuracy, accuracies = calculate_accuracy(lstm_model, category_lines, all_categories)
    print(f'Accuracy: {accuracy * 100:.2f}%')
    
    # 绘制准确率变化曲线
    plt.figure(figsize=(10, 5))
    plt.plot(accuracies)
    plt.title('Accuracy Over Time (LSTM)')
    plt.xlabel('Number of Samples')
    plt.ylabel('Accuracy')
    plt.show()

import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from models.rnn_seq2seq import Seq2Seq, Encoder, Decoder
from models.attention_seq2seq import AttentionSeq2Seq, EncoderRNN, AttentionDecoder
from utils import *
import os

def train(model_type='basic', n_epochs=10, learning_rate=0.01, 
          print_every=100, plot_every=100, hidden_size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 准备数据
    input_lang, output_lang, pairs = prepareData('eng', 'fra', max_length=10)
    
    # 创建保存目录
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # 初始化模型
    if model_type == 'basic':
        encoder = Encoder(input_lang.n_words, hidden_size).to(device)
        decoder = Decoder(hidden_size, output_lang.n_words).to(device)
        model = Seq2Seq(encoder, decoder, device).to(device)
        model_save_path = 'checkpoints/basic_seq2seq.pt'
        results_save_path = 'results/basic_results.pkl'
    else:
        encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
        decoder = AttentionDecoder(hidden_size, output_lang.n_words).to(device)
        model = AttentionSeq2Seq(encoder, decoder, device).to(device)
        model_save_path = 'checkpoints/attention_seq2seq.pt'
        results_save_path = 'results/attention_results.pkl'
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()
    
    # 训练过程中记录的数据
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0
    training_pairs = [tensorsFromPair(random.choice(pairs), input_lang, output_lang, device)
                     for _ in range(n_epochs)]
    
    start = time.time()
    
    for epoch in range(1, n_epochs + 1):
        for iter in range(1, len(training_pairs) + 1):
            training_pair = training_pairs[iter - 1]
            input_tensor = training_pair[0]
            target_tensor = training_pair[1]
            
            loss = train_iter(input_tensor, target_tensor, model, optimizer, criterion)
            print_loss_total += loss
            plot_loss_total += loss
            
            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print(f'%d %d%% %.4f' % (iter, iter / len(training_pairs) * 100, print_loss_avg))
            
            if iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0
        
        # 每个epoch结束后保存模型
        save_checkpoint(model, optimizer, epoch, plot_losses[-1], model_save_path)
    
    # 保存训练结果
    results = {
        'losses': plot_losses,
        'training_time': time.time() - start
    }
    save_results(results, results_save_path)
    
    return model, plot_losses

def train_iter(input_tensor, target_tensor, model, optimizer, criterion):
    optimizer.zero_grad()
    loss = 0
    
    output = model(input_tensor, target_tensor)
    
    target_length = target_tensor.size(0)
    for t in range(1, target_length):
        loss += criterion(output[t], target_tensor[t])
    
    loss.backward()
    optimizer.step()
    
    return loss.item() / target_length

def evaluate(model, input_lang, output_lang, sentence, device, max_length=10):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence, device)
        
        encoder_hidden = model.encoder.init_hidden(device)
        encoder_output, encoder_hidden = model.encoder(input_tensor[0], encoder_hidden)
        
        decoder_input = torch.tensor([[0]], device=device)  # SOS
        decoder_hidden = encoder_hidden
        
        decoded_words = []
        
        for _ in range(max_length):
            decoder_output, decoder_hidden = model.decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            if topi.item() == 1:  # EOS
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])
            
            decoder_input = topi.squeeze().detach()
        
        return decoded_words

def compare_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 训练基础Seq2Seq模型
    print("Training basic Seq2Seq model...")
    basic_model, basic_losses = train(model_type='basic')
    
    # 训练注意力机制Seq2Seq模型
    print("\nTraining attention Seq2Seq model...")
    attention_model, attention_losses = train(model_type='attention')
    
    # 绘制损失对比图
    plt.figure(figsize=(10, 6))
    plt.plot(basic_losses, label='Basic Seq2Seq')
    plt.plot(attention_losses, label='Attention Seq2Seq')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.savefig('results/loss_comparison.png')
    plt.close()
    
    # 保存比较结果
    comparison_results = {
        'basic_losses': basic_losses,
        'attention_losses': attention_losses,
        'basic_final_loss': basic_losses[-1],
        'attention_final_loss': attention_losses[-1]
    }
    save_results(comparison_results, 'results/comparison_results.pkl')

if __name__ == "__main__":
    compare_models()

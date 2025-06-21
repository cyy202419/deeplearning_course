import torch
import jieba 
from models.Transformer import Transformer
from Trainer.maskfunc import make_src_mask, make_trg_mask
import numpy as np

class ChineseTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
    
    def tokenizer(self, sentence):
        tokens = jieba.lcut(sentence)
        return [token if token in self.vocab.word2id else '<unk>' for token in tokens]

class Translator:
    def __init__(self, model_path, zh_vocab, en_vocab, config, device):
        self.model = self._load_model(model_path, config, device)
        self.zh_vocab = zh_vocab
        self.en_vocab = en_vocab
        self.device = device
        self.zh_tokenizer = ChineseTokenizer(zh_vocab)
        
    def _load_model(self, model_path, config, device):
        model = Transformer(**config).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        model.eval()
        return model

    # 翻译任意输入的中文句子
    def translate(self, sentence, max_len=100):
        #使用jieba分词
        tokens = self.zh_tokenizer.tokenizer(sentence)
        tokens = ['<sos>'] + tokens + ['<eos>']
        return self.translate_line(tokens, max_len)
    
    #翻译中文文本文件，并且将翻译结果保存到指定路径
    def translate_file(self, filename, output_path='translation.txt', max_len=100):
        
        src_lines = self.readfile(filename)
        
        for line in src_lines:
            translation = self.translate_line(line, max_len)
            self.save_translation(line, translation, output_path)

    #翻译单行中文文本
    def translate_line(self, tokens, max_len=100):
        #转换为词ID序列
        tokens = [self.zh_vocab.word2id.get(word, self.zh_vocab.word2id['<unk>']) for word in tokens]
        src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(self.device) 
        #生成源语言掩码
        src_mask = make_src_mask(src_tensor, self.zh_vocab, self.device)
        
        with torch.no_grad():
            #编码器处理
            enc_src = self.model.encoder(src_tensor, src_mask)
            
            #初始化目标序列 (只包含<sos>)
            trg = [self.en_vocab.word2id['<sos>']]
            
            #自回归生成
            for i in range(max_len):
                trg_tensor = torch.LongTensor(trg).unsqueeze(0).to(self.device)
                trg_mask = make_trg_mask(trg_tensor, self.en_vocab, self.device)
                
                #解码器预测
                with torch.no_grad():
                    output = self.model.decoder(trg_tensor, enc_src, src_mask, trg_mask)
                    output = self.model.fc(output)  # 线性层映射到词汇表
                
                #获取最新预测词
                pred_token = output.argmax(2)[:, -1].item()
                trg.append(pred_token)
                
                #遇到<eos>则停止
                if pred_token == self.en_vocab.word2id['<eos>']:
                    break
        
        #转换为英文单词序列
        trg_tokens = [self.en_vocab.id2word.get(idx, '<unk>') for idx in trg]
        return ' '.join(trg_tokens[1:-1])  # 去掉开头的<sos>

    # 读取并处理输入的中文文本
    def readfile(self, filename):
        """读取训练数据"""
        with open(filename, 'r', encoding='utf-8') as f:
            src_lines = [line.strip() for line in f.readlines() if line.strip()]

        src_lines = [['<sos>'] + line.strip().split() + ['<eos>'] for line in src_lines]
        return src_lines
    
    # 保存翻译结果到指定文件
    def save_translation(self, original_text, translated_text, output_path='translations.txt'):
            try:
                with open(output_path, 'a', encoding='utf-8') as f:
                    f.write(f"原文: {original_text}\n")
                    f.write(f"翻译: {translated_text}\n")
                    f.write("-" * 50 + "\n")  # 分隔线
                print(f"翻译已成功保存到 {output_path}")
            except Exception as e:
                print(f"保存文件时出错: {e}")

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import json


class Vocabulary:
    def __init__(self, name, min_freq=1):
        self.name = name
        self.word2id = {'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3}
        self.id2word = {0: '<pad>', 1: '<unk>', 2: '<sos>', 3: '<eos>'}
        self.word2count = {}
        self.min_freq = min_freq
        self.n_words = 4  # Count special tokens
    
    def add_word(self, word):
        if word not in self.word2count:
            self.word2count[word] = 1
        else:
            self.word2count[word] += 1

    def __getitem__(self, token):
        return self.word2id.get(token, self.word2id['<unk>'])
    
    
    def build_vocab(self):
        for word, count in self.word2count.items():
            if count >= self.min_freq:
                self.word2id[word] = self.n_words
                self.id2word[self.n_words] = word
                self.n_words += 1
    
    def save(self, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        data = {
            'name': self.name,
            'word2id': self.word2id,
            'id2word': self.id2word,
            'word2count': self.word2count,
            'min_freq': self.min_freq,
            'n_words': self.n_words
        }
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        vocab = cls(data['name'], data['min_freq'])
        vocab.word2id = data['word2id']
        vocab.id2word = {int(k): v for k, v in data['id2word'].items()}
        vocab.word2count = data['word2count']
        vocab.n_words = data['n_words']
        return vocab

class Zh2EnDataset(Dataset):
    def __init__(self, src_lines, trg_lines, src_vocab, trg_vocab):
        self.src_lines = src_lines
        self.trg_lines = trg_lines
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
    
    def __len__(self):
        return len(self.src_lines)
    
    def __getitem__(self, index):
        src_tokens = self.src_lines[index]
        trg_tokens = self.trg_lines[index]

        # 转换为 ID 序列
        src_ids = [
            self.src_vocab.word2id[word] if word in self.src_vocab.word2id 
            else self.src_vocab.word2id['<unk>'] 
            for word in src_tokens
        ]
        
        trg_ids = [
            self.trg_vocab.word2id[word] if word in self.trg_vocab.word2id 
            else self.trg_vocab.word2id['<unk>'] 
            for word in trg_tokens
        ]
        
        return torch.LongTensor(src_ids), torch.LongTensor(trg_ids)

class Zh2EnDataLoader:
    def __init__(self, src_train_file, trg_train_file, valid_file, 
                 src_vocab_file, trg_vocab_file, batch_size, 
                 shuffle=True, min_freq=1, rebuild_vocab=False):
        self.src_train_file = src_train_file
        self.trg_train_file = trg_train_file
        self.valid_file = valid_file
        self.src_vocab_file = src_vocab_file
        self.trg_vocab_file = trg_vocab_file
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.min_freq = min_freq
        
        # Build or load vocabularies
        if rebuild_vocab or not os.path.exists(src_vocab_file):
            self._build_vocabularies()
        self.src_vocab = Vocabulary.load(src_vocab_file)
        self.trg_vocab = Vocabulary.load(trg_vocab_file)
    
    def _build_vocabularies(self):
        # Build Chinese vocabulary
        zh_vocab = Vocabulary('zh', self.min_freq)
        with open(self.src_train_file, 'r', encoding='utf-8') as f:
            for line in f:
                for word in line.strip().split():
                    zh_vocab.add_word(word)
        zh_vocab.build_vocab()
        zh_vocab.save(self.src_vocab_file)
        
        # Build English vocabulary
        en_vocab = Vocabulary('en', self.min_freq)
        with open(self.trg_train_file, 'r', encoding='utf-8') as f:
            for line in f:
                for word in line.strip().split():
                    en_vocab.add_word(word)
        en_vocab.build_vocab()
        en_vocab.save(self.trg_vocab_file)
    
    def _read_train_data(self):
        """读取训练数据"""
        with open(self.src_train_file, 'r', encoding='utf-8') as f:
            src_lines = np.array(f.readlines())
        with open(self.trg_train_file, 'r', encoding='utf-8') as f:
            trg_lines = np.array(f.readlines())
        
        assert len(src_lines) == len(trg_lines), "训练文件行数不匹配"
        if self.shuffle:
            idx = np.random.permutation(len(src_lines))
            src_lines = src_lines[idx]
            trg_lines = trg_lines[idx]
            
        return self._preprocess_data(src_lines, trg_lines)
    
    def _read_valid_data(self):
        """读取验证数据"""
        with open(self.valid_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        
        # 验证集格式是中文-英文交替出现
        src_lines = []
        trg_lines = []
        for i in range(0, len(lines), 2):
            src_lines.append(lines[i])
            trg_lines.append(lines[i+1])
        
        return self._preprocess_data(src_lines, trg_lines)
        
    def _preprocess_data(self, src_lines, trg_lines):
        """预处理（添加特殊标记并分批）"""
        src_lines = [['<sos>'] + line.strip().split() + ['<eos>'] for line in src_lines]
        trg_lines = [['<sos>'] + line.strip().split() + ['<eos>'] for line in trg_lines]

        return src_lines, trg_lines
    
    def get_dataloaders(self):
        """获取训练和验证数据加载器"""
        train_src, train_trg = self._read_train_data()
        valid_src, valid_trg = self._read_valid_data()
        
        train_dataset = Zh2EnDataset(train_src, train_trg, self.src_vocab, self.trg_vocab)
        valid_dataset = Zh2EnDataset(valid_src, valid_trg, self.src_vocab, self.trg_vocab)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=4, pin_memory=True, persistent_workers=True, shuffle=self.shuffle, 
                                collate_fn=self.collate_fn)
        valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, num_workers=4, pin_memory=True, persistent_workers=True, shuffle=self.shuffle, 
                                collate_fn=self.collate_fn)
        
        return train_loader, valid_loader
    
    def collate_fn(self, batch):
        # batch 是一个列表，每个元素是 (src_ids, trg_ids)
        src_batch, trg_batch = zip(*batch)
        
        # 计算 batch 内最大长度
        src_len = max(len(src) for src in src_batch)
        trg_len = max(len(trg) for trg in trg_batch)
        
        # 填充到相同长度
        src_padded = torch.LongTensor(len(batch), src_len).fill_(self.src_vocab.word2id['<pad>'])
        trg_padded = torch.LongTensor(len(batch), trg_len).fill_(self.trg_vocab.word2id['<pad>'])
        
        for i, (src, trg) in enumerate(zip(src_batch, trg_batch)):
            src_padded[i, :len(src)] = src
            trg_padded[i, :len(trg)] = trg
        
        return src_padded, trg_padded

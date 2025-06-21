import torch
from torch.nn import functional as F
import os
import logging
import time
import math
from .maskfunc import make_src_mask, make_trg_mask
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

class BaseTrainer:
    def __init__(self, dirpath='checkpoint'):
        self.dirpath = dirpath
        os.makedirs(dirpath, exist_ok=True)
    
    #def save(self, model, optimizer, acc, epoch):
    #    ckpt_path = f"{self.dirpath}/epoch{epoch}_acc{acc:.2f}_"
    #    filename = time.strftime(ckpt_path + '%m%d_%H_%M.pth')
    #    torch.save({
    #        'model_state': model.state_dict(),
    #    #    'optimizer_state': optimizer.state_dict()
    #    }, filename)
    """def save_model(self, model, train_loss, acc, bleu, epoch, is_valid=False, is_final=False, is_bleu=False):
    if not is_final:
        if is_valid:
            ckpt_path = f"{self.dirpath}/best_valid_epoch{epoch}_loss{train_loss:.2f}_acc{acc:.2f}.pth"
        else:
            ckpt_path = f"{self.dirpath}/best_train_epoch{epoch}_loss{train_loss:.2f}_acc{acc:.2f}.pth"
    else:
        ckpt_path = f"{self.dirpath}/final_epoch{epoch}_loss{train_loss:.2f}_acc{acc:.2f}.pth"
        
    torch.save({'model_state': model.state_dict()}, ckpt_path)
    return ckpt_path
    """
    
    
    def save_model(self, model, ckpt_path):
        torch.save({'model_state': model.state_dict()}, ckpt_path)
        
    def setup_logger(name):
        os.makedirs("log", exist_ok=True)
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        
        file_handler = logging.FileHandler(f'log/{name}.log')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        
        console_handler = logging.StreamHandler()
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        return logger


class Trainer(BaseTrainer):
    def __init__(self, model, data_loader, valid_data_loader, optimizer, criterion,
                 src_vocab, trg_vocab, 
                 device='cuda' if torch.cuda.is_available() else 'cpu',      
                 loggername='train_log',
                 warmup_epochs=10, total_epochs=50, initial_lr=0.001, min_lr=1e-6):
        super().__init__()
        self.model = model
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.logger = BaseTrainer.setup_logger(loggername)
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        
        self.best_val_loss = float('inf')  # 初始化最优验证损失为无穷大
        self.best_train_loss = float('inf')  # 初始化最优训练损失为无穷大
        self.best_bleu = 0.0  # 初始化最优BLEU分数为0.0
        self.best_train_path = None  # 初始化最优训练模型路径为None
        self.best_val_path = None  # 初始化最优验证模型路径为None
        self.best_bleu_path = None  # 初始化最优BLEU模型路径为None
        
        
        
    def _update_lr(self, epoch):
        """基于epoch的学习率更新"""
        if epoch < self.warmup_epochs:
            lr = self.initial_lr * ((epoch + 1) / self.warmup_epochs)**1.5
        else:
            # 余弦退火
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5*(self.initial_lr-self.min_lr)*(1 + math.cos(math.pi*progress))
        
        # 更新所有参数组的学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def train(self):
        patience = 5
        val_counter = 0
        bleu_counter = 0
        for epoch in range(self.total_epochs):
            self._update_lr(epoch)
            train_loss = self._train_epoch()
            val_loss = self._valid_epoch()
            bleu = self.bleu4_score()
            self.logger.info(
                f'Epoch {epoch}: LR={self.optimizer.param_groups[0]["lr"]:.2e}, '
                f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, BlEU-4: {bleu:.2f}'
                )
            
            #self.logger.info(
            #    f'Epoch {epoch}: LR={self.optimizer.param_groups[0]["lr"]:.2e}, '
            #    f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}'
            #    )
             
            if epoch > self.warmup_epochs: 
                if train_loss < self.best_train_loss:
                    self.best_train_loss = train_loss
                    # 删除旧的训练最佳模型（如果有）
                    if self.best_train_path and os.path.exists(self.best_train_path):
                        os.remove(self.best_train_path)
                    self.best_train_path = f"{self.dirpath}/best_train_epoch{epoch}_loss{train_loss:.2f}_val_loss{val_loss:.2f}_bleu_{bleu:.2f}.pth" 
                    self.save_model(self.model, self.best_train_path)

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    val_counter = 0
                    # 删除旧的验证最佳模型（如果有）
                    if self.best_val_path and os.path.exists(self.best_val_path):
                        os.remove(self.best_val_path)
                    self.best_val_path = f"{self.dirpath}/best_valid_epoch{epoch}_loss{train_loss:.2f}_val_loss{val_loss:.2f}_bleu_{bleu:.2f}.pth"
                    self.save_model(self.model, self.best_val_path)
                else:
                    val_counter += 1
                    
                if bleu > self.best_bleu:
                    self.best_bleu = bleu
                    bleu_counter = 0
                    # 删除旧的BLEU最佳模型（如果有）
                    if self.best_bleu_path and os.path.exists(self.best_bleu_path):
                        os.remove(self.best_bleu_path)
                    self.best_bleu_path = f"{self.dirpath}/best_bleu_epoch{epoch}_loss{train_loss:.2f}_val_loss{val_loss:.2f}_bleu_{bleu:.2f}.pth"
                    self.save_model(self.model, self.best_bleu_path)
                else:
                    bleu_counter += 1
                    
                # 共同控制训练退出
                if val_counter >= patience and bleu_counter >= patience:
                    self.logger.info(f"Early stopping at epoch {epoch} due to no improvement in validation loss and BLEU score.")
                    break

            
            if epoch == self.total_epochs - 1:
               self.save_model(self.model, f"{self.dirpath}/final{epoch}_loss{train_loss:.2f}_val_loss{val_loss:.2f}_bleu_{bleu:.2f}.pth" )
                      

    def _train_epoch(self):
        self.model.train()  # 设置模型为训练模式
        total_loss = 0
        
        for _, (src, trg) in enumerate(self.data_loader):
            # 数据移动到设备
            src = src.to(self.device)
            trg = trg.to(self.device)

            # 生成掩码
            src_mask = make_src_mask(src, self.src_vocab, self.device)
            trg_mask = make_trg_mask(trg[:, :-1], self.trg_vocab, self.device)

            # 梯度清零
            self.optimizer.zero_grad()
            
            # 前向传播
            output = self.model(src, trg[:, :-1], src_mask, trg_mask)
        
            # 调整输出形状
            # 原始输出: [batch_size, target_len-1, target_vocab_size]
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)

            # 调整目标形状
            # 目标: [batch_size, target_len] → 去掉<eos>后的展平
            trg = trg[:, 1:].contiguous().view(-1)

            # 计算损失
            loss = self.criterion(output, trg)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪（可调整参数1为其他值）
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            
            # 优化器更新
            self.optimizer.step()
            
            # 累计损失
            total_loss += loss.item()
        
        return total_loss



    def _valid_epoch(self):
        self.model.eval()  # 设置模型为评估模式
        val_loss = 0
        
        with torch.no_grad():  # 禁用梯度计算
            for _, (src, trg) in enumerate(self.valid_data_loader):
                # 数据移动到设备
                src = src.to(self.device)
                trg = trg.to(self.device)
                
                # 生成掩码
                src_mask = make_src_mask(src, self.src_vocab, self.device)
                trg_mask = make_trg_mask(trg[:, :-1], self.trg_vocab, self.device)
                
                # 前向传播 (去掉trg的最后一个token作为输入)
                output = self.model(src, trg[:, :-1], src_mask, trg_mask)
                output = F.log_softmax(output, dim=-1)  # 应用softmax
                
                # 调整输出形状 [batch_size, target_len-1, vocab_size] -> [batch_size*(target_len-1), vocab_size]
                output_dim = output.shape[-1]
                output = output.contiguous().view(-1, output_dim)
                
                # 调整目标形状 [batch_size, target_len] -> [batch_size*(target_len-1)] (去掉<sos>)
                trg = trg[:, 1:].contiguous().view(-1)
                
                # 计算损失
                val_loss += self.criterion(output, trg).item()
        
        # 返回平均验证损失
        return val_loss / len(self.valid_data_loader)
    
    
    def bleu4_score(self):
        self.model.eval()
        total_score = 0.0
        smoothie = SmoothingFunction().method4  # 使用平滑函数处理短句子
        
        with torch.no_grad():
            for _, (src, trg) in enumerate(self.valid_data_loader):
                # 数据移动到设备
                src = src.to(self.device)
                trg = trg.to(self.device)
                
                # 生成掩码
                src_mask = make_src_mask(src, self.src_vocab, self.device)
                trg_mask = make_trg_mask(trg[:, :-1], self.trg_vocab, self.device)
                
                # 前向传播
                output = self.model(src, trg[:, :-1], src_mask, trg_mask)
                output = F.log_softmax(output, dim=-1)
                
                # 获取预测结果 (batch_size, seq_len)
                preds = output.argmax(-1)
                
                # 处理每个样本
                for i in range(preds.size(0)):
                    # 获取参考译文 (去掉<sos>和<pad>)
                    reference = []
                    for idx in trg[i, 1:].tolist():  # 去掉<sos>
                        if idx == self.trg_vocab.word2id['<eos>'] or idx == self.trg_vocab.word2id['<pad>']:
                            break
                        reference.append(self.trg_vocab.id2word[idx])
                    
                    # 获取预测译文 (去掉<pad>和<eos>之后的token)
                    hypothesis = []
                    for idx in preds[i].tolist():
                        if idx == self.trg_vocab.word2id['<eos>'] or idx == self.trg_vocab.word2id['<pad>']:
                            break
                        hypothesis.append(self.trg_vocab.id2word[idx])
                    
                    # 计算BLEU-4分数
                    if len(reference) > 0 and len(hypothesis) > 0:
                        # 将参考译文和预测译文转换为单词列表
                        reference_tokens = [reference]  # BLEU需要references是列表的列表
                        hypothesis_tokens = hypothesis
                        
                        # 计算BLEU-4分数
                        score = sentence_bleu(
                            reference_tokens, 
                            hypothesis_tokens,
                            weights=(0.25, 0.25, 0.25, 0.25),  # BLEU-4权重
                            smoothing_function=smoothie
                        )
                        total_score += score
        
        # 返回平均BLEU-4分数
        return 100 * total_score / len(self.valid_data_loader.dataset)
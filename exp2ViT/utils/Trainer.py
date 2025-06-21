import time
import os
import logging
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
from tqdm import tqdm


class CheckpointManager:
    def __init__(self, dirpath='checkpoint'):
        self.dirpath = dirpath
        os.makedirs(dirpath, exist_ok=True)
    
    def save(self, model, optimizer, acc, epoch):
        ckpt_path = f"{self.dirpath}/ViT_epoch{epoch}_acc{acc:.2f}.pth"
        torch.save({
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict()
        }, ckpt_path)

def progress_bar(current, total, msg=None):
    pbar = tqdm(total=total, dynamic_ncols=True)
    pbar.set_description(msg)
    pbar.update(current)
    pbar.close()

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

class Trainer:
    def __init__(self, model, optimizer, criterion, device, epochs=1):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.epochs = epochs
        self.best_acc = 0
        self.checkpoint_manager = CheckpointManager()
        self.logger = setup_logger('ViT_trainer')
        
        self.scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=epochs,  # 周期长度=总epoch数
            eta_min=1e-6  # 最小学习率
        )

    
    def sparse_selection(self, sparsity=0.3):
        """随机屏蔽部分参数梯度"""
        with torch.no_grad():
            for param in self.model.parameters():
                if len(param.shape) > 1:
                    mask = (torch.rand_like(param) > sparsity).float()
                    param.grad.mul_(mask)

    def train_epoch(self, epoch, trainloader):
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.sparse_selection()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            """
            if batch_idx % 20 == 0:
                progress_bar(batch_idx, len(trainloader), 
                            'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                                train_loss/(batch_idx+1), 
                                100.*correct/total, 
                                correct, 
                                total
                            ))
            """

        return train_loss / (batch_idx + 1)

    def test(self, epoch, testloader):
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                """
                if batch_idx % 20 == 0:
                    progress_bar(batch_idx, len(testloader), 
                                'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                                    test_loss/(batch_idx+1), 
                                    100.*correct/total,
                                    correct, 
                                    total
                                ))
                """
                
        acc = 100.*correct/total
        return test_loss / len(testloader), acc

    def run(self, trainloader, testloader, epochs):
        for epoch in range(epochs):
            train_loss = self.train_epoch(epoch, trainloader)
            test_loss, acc = self.test(epoch, testloader)
            
            self.scheduler.step()
            
            self.logger.info(f"Epoch {epoch}, lr: {self.optimizer.param_groups[0]['lr']:.7f}, train loss: {train_loss:.3f}, val loss: {test_loss:.3f}, val acc: {acc:.2f}%")
            
            if acc > self.best_acc:
                self.best_acc = acc

            if (epoch+1) % 3 == 0:
                self.checkpoint_manager.save(self.model, self.optimizer, acc, epoch)
            
            #content = (f"{time.ctime()} Epoch [{epoch}], "
            #        f"lr: {self.optimizer.param_groups[0]['lr']:.7f}, "
            #        f"val loss: {test_loss:.3f}, val acc: {acc:.2f}%")
            #print(content)

        print(f"Training completed. Best accuracy: {self.best_acc:.2f}%")
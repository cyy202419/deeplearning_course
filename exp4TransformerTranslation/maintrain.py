import torch
import torch.nn as nn
from models.Transformer import Transformer
from Trainer.Trainer import Trainer
from Zh2EnDataLoader.Zh2EnDataLoader import Zh2EnDataLoader

def test_transformer_trainer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_loader = Zh2EnDataLoader(
        src_train_file='sample-submission-version/TM-training-set/chinese.txt',
        trg_train_file='sample-submission-version/TM-training-set/english.txt',
        valid_file='sample-submission-version/Dev-set/Niu.dev.txt',
        src_vocab_file='vocab/zh_vocab.json',
        trg_vocab_file='vocab/en_vocab.json',
        batch_size=128,
        shuffle=True,
        min_freq=1,
        rebuild_vocab=False)
    
    LITE_CONFIG = {
        'src_vocab_size': data_loader.src_vocab.n_words,
        'target_vocab_size': data_loader.trg_vocab.n_words,
        'h_dim': 768,
        'enc_pf_dim': 1024,
        'dec_pf_dim': 1024,
        'enc_n_layers': 4,
        'dec_n_layers': 4,
        'enc_n_heads': 8,
        'dec_n_heads': 8,
        'enc_dropout': 0.2,
        'dec_dropout': 0.2
    }

    # 获取数据加载器
    train_loader, valid_loader = data_loader.get_dataloaders()
    
    # Initialize model with lite config
    model = Transformer(**LITE_CONFIG).to(device)
    
    # Training components
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5) #无所谓定义多少lr
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)  # ignore padding
    
    # Create trainer
    trainer = Trainer(
        model=model,
        data_loader=train_loader,
        valid_data_loader=valid_loader,
        optimizer=optimizer,
        criterion=criterion,
        src_vocab=data_loader.src_vocab,
        trg_vocab=data_loader.trg_vocab,
        device=device,
        loggername='translationZh2En',
        warmup_epochs=10,
        total_epochs=100,
        initial_lr=3e-4,  # 实际控制学习率
        min_lr=1e-6,  # 最小学习率
    )
    
    
    # Run a few epochs
    print("Starting training test...")
    trainer.train()
    print("Training test completed successfully!\n")
    
if __name__ == "__main__":
    test_transformer_trainer()

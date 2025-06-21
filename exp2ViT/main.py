import torch
from data.dataloader import get_loader
from models.vit import *
import yaml
from utils.Trainer import Trainer

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)
    
def main(): 
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps") 
    config = load_config('configs/vit_cifar10_lite.yaml')
    trainloader, testloader = get_loader(config)


    model = ViT(
            image_size=config['model']['image_size'],
            patch_size=config['model']['patch_size'],
            num_classes=config['model']['num_classes'],
            dim=config['model']['dim'],
            depth=config['model']['depth'],
            heads=config['model']['heads'],
            mlp_dim=config['model']['mlp_dim'],
            dropout=config['model']['dropout'],
            emb_dropout=config['model']['emb_dropout'],
        ).to(device)
    
    
    # 加载预训练模型（可选）
    load_pretrained = True  # 设置为False跳过加载
    if load_pretrained:
        checkpoint_path = "checkpoint/ViT_epoch20_acc81.88.pth"
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        print(f"Loaded pretrained model from {checkpoint_path}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['lr'],
        betas=(0.9, 0.999),
        amsgrad=True,
        weight_decay=config['training']['weight_decay'],
        eps=1e-8,
    )

    criterion = torch.nn.CrossEntropyLoss()

    epochs = config['training']['epochs']
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=epochs,
    )

    # 启动训练
    trainer.run(trainloader, testloader, epochs)
    
    
    
if __name__ == "__main__":
    main()

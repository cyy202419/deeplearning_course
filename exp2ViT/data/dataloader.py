from torchvision import transforms
import torchvision
import torch

def get_loader(config):
    """
    trans_train = transforms.Compose([
    transforms.RandomResizedCrop(224),  # 随机裁剪+缩放至224x224
    transforms.RandomHorizontalFlip(),  # 50%概率水平翻转
    transforms.ToTensor(),              # 转Tensor并归一化[0,1]
    transforms.Normalize(              # ImageNet标准归一化
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
    ])

    trans_valid = transforms.Compose([
        transforms.Resize(256),            # 短边缩放到256
        transforms.CenterCrop(224),        # 中心裁剪224x224
        transforms.ToTensor(),
        transforms.Normalize(              # 同训练集的归一化参数
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])
    """
    
    # CIFAR-10数据集预处理
    trans_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])

    trans_valid = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])

    # 训练集
    trainset = torchvision.datasets.CIFAR10(
        root=config['data']['root'], 
        train=True,
        download=True,          # 建议添加自动下载
        transform=trans_train)
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],)

    # 测试集
    testset = torchvision.datasets.CIFAR10(
        root=config['data']['root'],
        train=False,
        download=False,
        transform=trans_valid)

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],)
    
    return trainloader, testloader
import torch

def make_src_mask(src, src_vocab, device):
    # 创建源序列的掩码
    src_mask = (src != src_vocab['<pad>']).unsqueeze(1).unsqueeze(2)
    return src_mask.to(device)

def make_trg_mask(trg, trg_vocab, device):
    # 创建目标序列的掩码
    trg_pad_mask = (trg != trg_vocab['<pad>']).unsqueeze(1).unsqueeze(2)
    trg_subsequent_mask = torch.tril(torch.ones((1, trg.shape[1], trg.shape[1]), device=device)).bool()
    trg_mask = trg_pad_mask & trg_subsequent_mask
    return trg_mask
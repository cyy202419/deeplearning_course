import torch
from Zh2EnDataLoader.Zh2EnDataLoader import Zh2EnDataLoader
from Translator.Translator import Translator

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

####################Best bleu model####################
translator = Translator(
        model_path='checkpoint/best_bleu_epoch35_loss2164.93_val_loss4.26_bleu_18.20.pth',
        zh_vocab=data_loader.src_vocab,
        en_vocab=data_loader.trg_vocab,
        config=LITE_CONFIG,
        device=device
    )

chinese_sentence = "民主法制建设取得明显进展，依法治国迈出坚实的步伐"
translation = translator.translate(chinese_sentence)
translator.save_translation(chinese_sentence, translation)

chinese_sentence = "金正恩批评称这是由于疏忽和不负责任，不尊重科学的经验主义而产生的无法容忍的重大事故和犯罪行为。" 
translation = translator.translate(chinese_sentence)
translator.save_translation(chinese_sentence, translation)

chinese_sentence = "一系列优化政策落地丰富了消费场景，缓解了退税商店偏少、分布不均等问题。"
translation = translator.translate(chinese_sentence)
translator.save_translation(chinese_sentence, translation)

####################Best valid model####################
translator = Translator(
        model_path='checkpoint/best_valid_epoch22_loss2683.22_val_loss4.00_bleu_16.96.pth',
        zh_vocab=data_loader.src_vocab,
        en_vocab=data_loader.trg_vocab,
        config=LITE_CONFIG,
        device=device
    )

chinese_sentence = "民主法制建设取得明显进展，依法治国迈出坚实的步伐"
translation = translator.translate(chinese_sentence)
translator.save_translation(chinese_sentence, translation)

chinese_sentence = "金正恩批评称这是由于疏忽和不负责任，不尊重科学的经验主义而产生的无法容忍的重大事故和犯罪行为。" 
translation = translator.translate(chinese_sentence)
translator.save_translation(chinese_sentence, translation)

chinese_sentence = "一系列优化政策落地丰富了消费场景，缓解了退税商店偏少、分布不均等问题。"
translation = translator.translate(chinese_sentence)
translator.save_translation(chinese_sentence, translation)


####################Best train model####################
translator = Translator(
        model_path='checkpoint/best_train_epoch40_loss2049.56_val_loss4.36_bleu_17.65.pth',
        zh_vocab=data_loader.src_vocab,
        en_vocab=data_loader.trg_vocab,
        config=LITE_CONFIG,
        device=device
    )

chinese_sentence = "民主法制建设取得明显进展，依法治国迈出坚实的步伐"
translation = translator.translate(chinese_sentence)
translator.save_translation(chinese_sentence, translation)

chinese_sentence = "金正恩批评称这是由于疏忽和不负责任，不尊重科学的经验主义而产生的无法容忍的重大事故和犯罪行为。" 
translation = translator.translate(chinese_sentence)
translator.save_translation(chinese_sentence, translation)

chinese_sentence = "一系列优化政策落地丰富了消费场景，缓解了退税商店偏少、分布不均等问题。"
translation = translator.translate(chinese_sentence)
translator.save_translation(chinese_sentence, translation)

# Translate the test set file
translator.translate_file('sample-submission-version/Test-set/Niu.test.txt', 'Niu.test.translation.txt')
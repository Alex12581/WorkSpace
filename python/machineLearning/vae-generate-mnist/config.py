'''
vae keras
'''

class Config(object):
    
    latent_dim = 8
    epochs = 2000
    learning_rate = 0.001  # 学习率
    batch_size = 20  # 批数据量
    patience = 10  # loss 连续 patience 次没有有效的下降则停止训练
    min_delta = 0.0001 # loss 减少了 min_delta 以上才认为是有效的

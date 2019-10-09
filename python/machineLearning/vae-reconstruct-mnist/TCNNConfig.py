'''
vae keras
'''

class TCNNConfig(object):
    
    epochs = 300
    latent_dim = 50
    # layers = [200, 150, 100]  # 堆栈每一层的隐藏层数
    # learning_rate = [1e-3, 1e-6, 1e-6]  # 学习率
    # training_epochs = [300, 100, 100]  # 训练次数
    batch_size = 20  # 批数据量
    # corruption_level = 0  # 噪音比例
    # step = 20  # 间隔多少 epoch 输出结果一次
    ceiling = 15849  # 加载文本数量上限（0-15849）
    max_features = 3000  # 特征值数量
    clusters = 8  # 聚类的数量

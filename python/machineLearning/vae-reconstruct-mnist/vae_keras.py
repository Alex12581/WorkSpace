# -*- coding: utf-8 -*-

from keras.layers import Input, Dense, Lambda, Dropout
from keras.models import Model
from keras import backend as K
from keras import metrics

class VAE(object):
    def __init__(self, input_dim, latent_dim):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.build_model()
    
    def build_model(self):
        x = Input(shape=(self.input_dim,))
        # 算p(Z|X)的均值和方差
        z_mean = Dense(self.latent_dim)(x)
        z_log_var = Dense(self.latent_dim)(x)
        # 重参数层，相当于给输入加入噪声
        z = Lambda(self._sampling, output_shape=(self.latent_dim,))([z_mean, z_log_var])
        # 解码层，也就是生成器部分
        # z_drop = Dropout(0.5)(z)
        x_decoded_mean = Dense(self.input_dim, activation='sigmoid')(z)

        # 建立模型
        self.vae = Model(x, x_decoded_mean)
        self.encoder = Model(x, z)
        
        # xent_loss是重构loss，kl_loss是KL loss
        xent_loss = self.input_dim * metrics.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        vae_loss = K.mean(xent_loss + kl_loss)
        
        # add_loss是新增的方法，用于更灵活地添加各种loss
        self.vae.add_loss(vae_loss)
        self.vae.compile(optimizer='rmsprop')
    
    def _sampling(self, args):
        '''重参数技巧'''
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.latent_dim), mean=0.,
                                  stddev=1.0)
        return z_mean + K.exp(z_log_var / 2) * epsilon
    
    def fit(self, X_train, batch_size=100, epochs=50):
        # self.vae.fit(x=X_train, y=X_train, shuffle=True, epochs=epochs, batch_size=batch_size)
        self.vae.fit(X_train, shuffle=True, epochs=epochs, batch_size=batch_size)

    def project(self, X_test, batch_size=100):
        return self.encoder.predict(X_test, batch_size=batch_size)

    def reconstruct(self, X_test, batch_size=100):
        return self.vae.predict(X_test, batch_size=batch_size)

# -*- coding: utf-8 -*-

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import optimizers
from keras import backend as K
from keras import metrics
from config import Config as config
from keras.callbacks import *

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
        x_decoded_mean = Dense(self.input_dim, activation='sigmoid')(z)

        # 建立模型
        self.vae = Model(x, x_decoded_mean)
        self.encoder = Model(x, z)

        #
        z_input = Input(shape=(self.latent_dim,))
        generated = self.vae.layers[-1](z_input)

        self.decoder = Model(z_input, generated)
        
        # xent_loss是重构loss，kl_loss是KL loss
        # xent_loss = self.input_dim * metrics.binary_crossentropy(x, x_decoded_mean)
        xent_loss = self.input_dim * metrics.categorical_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        vae_loss = K.mean(xent_loss + kl_loss)
        
        # add_loss是新增的方法，用于更灵活地添加各种loss
        self.vae.add_loss(vae_loss)
        self.vae.compile(optimizer=optimizers.RMSprop(lr=config.learning_rate), loss=None)  # 默认学习率 0.001
        # self.vae.compile(optimizer=optimizers.Adadelta(lr=1.0), loss=None)  # 默认学习率 1.0

        self.z_mean_model = Model(x, z_mean)
        self.z_log_var_model = Model(x, z_log_var)
    
    def _sampling(self, args):
        '''重参数技巧'''
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.latent_dim), mean=0.,
                                  stddev=1.0)
        return z_mean + K.exp(z_log_var / 2) * epsilon
    
    def fit(self, X_train, batch_size=config.batch_size, epochs=config.epochs):
        return self.vae.fit(x=X_train, y=None, shuffle=True, epochs=epochs, batch_size=batch_size, callbacks=[EarlyStopping(monitor='loss', patience=config.patience, verbose=1, mode='min', restore_best_weights=True, min_delta=config.min_delta)])

    def project(self, X_test, batch_size=config.batch_size):
        return self.encoder.predict(X_test, batch_size=batch_size)

    def reconstruct(self, X_test, batch_size=config.batch_size):
        return self.vae.predict(X_test, batch_size=batch_size)

    def generate(self, z, batch_size=config.batch_size):
        return self.decoder.predict(z, batch_size=batch_size)

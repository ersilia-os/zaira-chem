#!/usr/bin/env python
# -*- coding: utf-8 -*-

import keras.backend as kb
from keras.models import Sequential
from keras.layers import Conv1D, Dense, Dropout, LSTM
from FCD import build_masked_loss


def build_chemblnet_model(max_len=350, chars=35, n_targs=2668):
    model = Sequential([
        Dropout(0.1, input_shape=(max_len, chars), name='Dropout_1'),
        Conv1D(32, (4,), strides=(2,), activation='selu', padding='same', kernel_initializer='VarianceScaling', name='Conv_1'),
        Dropout(0.5, name='Dropout_2'),
        Conv1D(32, (4,), strides=(2,), activation='selu', padding='same', kernel_initializer='VarianceScaling', name='Conv_2'),
        Dropout(0.5, name='Dropout_3'),
        LSTM(128, activation='tanh', kernel_initializer='VarianceScaling', return_sequences=True, name='LSTM_1'),
        Dropout(0.1, name='Dropout_4'),
        LSTM(512, activation='tanh', kernel_initializer='VarianceScaling', name='LSTM_2'),
        Dense(n_targs, activation='sigmoid', kernel_initializer='VarianceScaling', name='Dense')
    ])

    model.compile(optimizer='adam', loss=build_masked_loss(kb.binary_crossentropy, 0.5))
    return model

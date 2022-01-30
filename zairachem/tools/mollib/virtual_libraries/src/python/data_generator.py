##########################################
# Implementation of a generator based on
# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html
##########################################

import os, sys
import numpy as np
import pandas as pd
import keras


class DataGenerator(keras.utils.Sequence):
    "Generates data for Keras"

    def __init__(
        self,
        list_IDs,
        batch_size,
        max_len_model,
        path_data,
        n_chars,
        indices_token,
        token_indices,
        pad_char,
        start_char,
        end_char,
        shuffle=True,
    ):
        "Initialization"
        self.max_len_model = max_len_model
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.path_data = path_data
        self.n_chars = n_chars

        self.pad_char = pad_char
        self.start_char = start_char
        self.end_char = end_char

        self.on_epoch_end()

        f = open(self.path_data)
        self.lines = f.readlines()

        self.indices_token = indices_token
        self.token_indices = token_indices

    def __len__(self):
        "Denotes the number of batches per epoch"
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        "Generate one batch of data"
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def one_hot_encode(self, token_list, n_chars):
        output = np.zeros((token_list.shape[0], n_chars))
        for j, token in enumerate(token_list):
            output[j, token] = 1
        return output

    def smi_to_int(self, smile):
        """
        this will turn a list of smiles in string format
        and turn them into a np array of int, with padding
        """
        token_list = self.tokenize_sequence(smile)
        self.pad_seqs(token_list)
        int_list = [int(x[0]) for x in token_list]

        return np.asarray(int_list)

    def tokenize_sequence(self, smile):
        """
        Tokenizes a list of sequences into a list of token lists
        """

        token_lists = []
        for x in smile:
            mol_tokens = []
            posit = 0
            while posit < len(x):
                t, p = self.get_token(x, posit)
                posit = p
                mol_tokens += [self.token_indices[t]]
            token_lists.append(mol_tokens)
        return token_lists

    def get_token(self, text, position):
        """
        Return token from text at a particular position, assumes given position is valid
        """
        return list(text)[position], position + 1

    def pad_seqs(self, sequence):
        """
        Pad sequences to given length
        """
        padding = [self.token_indices[self.pad_char]] * (
            self.max_len_model - len(sequence)
        )
        padding_arr = [[x] for x in padding]
        sequence.extend(padding_arr)

    def enclose_smile(self, smi):
        """
        Add the start and end char.
        Used when we read smiles directly
        from the txt file
        """
        smi = self.start_char + smi + self.end_char
        return smi

    def clean_smile(self, smi):
        """remove return line symbols if present"""
        smi = smi.replace("\n", "")
        return smi

    def __data_generation(self, list_IDs_temp):
        "Generates batch of data containing batch_size samples"

        switch = 1
        y = np.empty(
            (self.batch_size, self.max_len_model - switch, self.n_chars), dtype=int
        )
        X = np.empty(
            (self.batch_size, self.max_len_model - switch, self.n_chars), dtype=int
        )

        for i, ID in enumerate(list_IDs_temp):
            smi = self.lines[ID]
            # remove return line symbols
            smi = self.clean_smile(smi)
            # add starting and ending char
            smi = self.enclose_smile(smi)
            data = self.smi_to_int(smi)

            X[i] = self.one_hot_encode(data[:-1], self.n_chars)
            y[i] = self.one_hot_encode(data[1:], self.n_chars)

        return X, y

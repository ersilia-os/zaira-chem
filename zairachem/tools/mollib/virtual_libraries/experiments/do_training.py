# Copyright (c) 2019 ETH Zurich

import os, sys
import argparse
import time
import configparser
import ast

from keras import backend as K
from keras.layers import Dense, LSTM, TimeDistributed, BatchNormalization
from keras.optimizers import Adam
from keras.models import Sequential, load_model, Model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import keras

sys.path.append("../src/")
from python import helper as hp
from python import data_generator as data_generator
from python import fixed_parameters as FP

parser = argparse.ArgumentParser(description="Run training")
parser.add_argument(
    "-fn",
    "--filename",
    type=str,
    help="Path to the fine-tuning txt file",
    required=True,
)
parser.add_argument("-v", "--verbose", type=bool, help="Verbose", required=True)


class SeqModel:
    """Class to define the language model, i.e the neural net"""

    def __init__(
        self, n_chars, max_length, layers, dropouts, trainables, lr, verbose=False
    ):

        self.n_chars = n_chars
        self.max_length = max_length

        self.layers = layers
        self.dropouts = dropouts
        self.trainables = trainables
        self.lr = lr

        self.model = None
        self.build_model()

    def build_model(self):

        self.model = Sequential()
        self.model.add(BatchNormalization(input_shape=(None, self.n_chars)))

        for neurons, dropout, trainable in zip(
            self.layers, self.dropouts, self.trainables
        ):
            self.model.add(
                LSTM(
                    neurons,
                    unit_forget_bias=True,
                    dropout=dropout,
                    trainable=trainable,
                    return_sequences=True,
                )
            )
        self.model.add(BatchNormalization())
        self.model.add(TimeDistributed(Dense(self.n_chars, activation="softmax")))

        optimizer = Adam(lr=self.lr)
        self.model.compile(loss="categorical_crossentropy", optimizer=optimizer)


def create_model_checkpoint(period, save_path):
    """Function to save the trained model during training"""
    filepath = save_path + "{epoch:02d}.h5"
    checkpointer = ModelCheckpoint(
        filepath=filepath, verbose=0, save_best_only=False, period=period
    )

    return checkpointer


if __name__ == "__main__":

    start = time.time()

    ####################################
    # get back parameters
    args = vars(parser.parse_args())

    verbose = args["verbose"]
    filename = args["filename"]
    name_data = filename.split("/")[-1].replace(".txt", "")
    config = configparser.ConfigParser()
    config.read("parameters.ini")

    # get back the experiment parameters
    min_len = int(config["PROCESSING"]["min_len"])
    max_len = int(config["PROCESSING"]["max_len"])
    mode = config["EXPERIMENTS"]["mode"]

    if verbose:
        print("\nSTART TRAINING")
    ####################################

    ####################################
    # Path to save the checkpoints
    save_path = f"results/{name_data}/models/"
    os.makedirs(save_path, exist_ok=True)
    ####################################

    ####################################
    # Neural net parameters
    patience_lr = int(config["MODEL"]["patience_lr"])
    batch_size = int(config["MODEL"]["batch_size"])
    epochs = int(config["MODEL"]["epochs"])
    period = int(config["MODEL"]["period"])
    n_workers = int(config["MODEL"]["n_workers"])
    min_lr = float(config["MODEL"]["min_lr"])
    factor = float(config["MODEL"]["factor"])
    ####################################

    ####################################
    # Generator parameters
    max_len_model = int(config["PROCESSING"]["max_len"]) + 2
    pad_char = FP.PROCESSING_FIXED["pad_char"]
    start_char = FP.PROCESSING_FIXED["start_char"]
    end_char = FP.PROCESSING_FIXED["end_char"]
    indices_token = FP.INDICES_TOKEN
    token_indices = FP.TOKEN_INDICES
    ####################################

    ####################################
    # Define monitoring
    monitor = "val_loss"
    lr_reduction = ReduceLROnPlateau(
        monitor=monitor, patience=patience_lr, verbose=0, factor=factor, min_lr=min_lr
    )
    ####################################

    ####################################
    # Path to the data
    augmentation = int(config["AUGMENTATION"][mode])
    dir_split_data = f"results/data/{name_data}/{min_len}_{max_len}_x{augmentation}/"
    if verbose:
        print(f"Data path : {dir_split_data}")

    # load partitions
    partition = {}
    path_partition_train = f"{dir_split_data}idx_tr"
    path_partition_valid = f"{dir_split_data}idx_val"

    partition["train"] = hp.load_obj(path_partition_train)
    partition["val"] = hp.load_obj(path_partition_valid)

    # get back the name of the training data from parameters
    path_data = f"{dir_split_data}{min_len}_{max_len}_x{augmentation}.txt"

    # finally, we infer the vocab size from the len
    # of the tokenization used
    vocab_size = len(indices_token)
    ####################################

    ####################################
    # Create the generators
    tr_generator = data_generator.DataGenerator(
        partition["train"],
        batch_size,
        max_len_model,
        path_data,
        vocab_size,
        indices_token,
        token_indices,
        pad_char,
        start_char,
        end_char,
        shuffle=True,
    )

    val_generator = data_generator.DataGenerator(
        partition["val"],
        batch_size,
        max_len_model,
        path_data,
        vocab_size,
        indices_token,
        token_indices,
        pad_char,
        start_char,
        end_char,
        shuffle=True,
    )
    ####################################

    ####################################
    # Create the checkpointer, the model and train.
    # Note: pretrained weights are loaded if we do
    # a fine-tuning experiment
    checkpointer = create_model_checkpoint(period, save_path)

    layers = ast.literal_eval(config["MODEL"]["neurons"])
    dropouts = ast.literal_eval(config["MODEL"]["dropouts"])
    trainables = ast.literal_eval(config["MODEL"]["trainables"])
    lr = float(config["MODEL"]["lr"])

    seqmodel = SeqModel(vocab_size, max_len_model, layers, dropouts, trainables, lr)

    if mode == "fine_tuning":
        # Load the pretrained model
        path_model = config["FINETUNING"]["path_model"]
        if path_model is None:
            raise ValueError(
                "You did not provide a path to a model to be loaded for fine-tuning"
            )

        pre_model = load_model(path_model)
        pre_weights = pre_model.get_weights()
        seqmodel.model.set_weights(pre_weights)

    if verbose:
        seqmodel.model.summary()

    history = seqmodel.model.fit_generator(
        generator=tr_generator,
        validation_data=val_generator,
        use_multiprocessing=True,
        epochs=epochs,
        callbacks=[checkpointer, lr_reduction],
        workers=n_workers,
    )

    # Save the loss history
    hp.save_obj(history.history, f"{save_path}history")

    end = time.time()
    print(f"TRAINING DONE in {end - start:.05} seconds")
    ####################################

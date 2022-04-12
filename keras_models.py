import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Dense, Dropout, Embedding, Flatten, BatchNormalization, Add
from tensorflow import keras
from tensorflow.keras.optimizers import RMSprop


class KerasModel:
    """Base class for training all Keras models"""

    def __init__(self, hyperparams: dict, binary: bool = False):
        self.model = None

        self.hyperparams = hyperparams

        self.optimizer = RMSprop(self.hyperparams["starting_lr"])

        self.feedforward_layer = self.create_feedforward_layers(hidden_units=[300, 100], dropout_rate=0.2)

        if binary:
            self.output_layer = Dense(1, activation="sigmoid")
            self.loss = "binary_crossentropy"
            self.metrics = ["binary_crossentropy"]
        else:
            self.output_layer = Dense(1)
            self.loss = "mae"
            self.metrics = ["mae"]

        self.callback_early_stopping = keras.callbacks.EarlyStopping(monitor=f"val_{self.metrics[0]}", patience=5,
                                                                     restore_best_weights=True)
        self.callback_decrease_lr = keras.callbacks.ReduceLROnPlateau(
            monitor=f"val_{self.metrics[0]}",
            factor=0.3,
            patience=2,
            min_lr=1e-6)

    @staticmethod
    def create_feedforward_layers(hidden_units, dropout_rate, name=None):
        fnn_layers = []

        for units in hidden_units:
            fnn_layers.append(Dropout(dropout_rate))
            fnn_layers.append(Dense(units, activation=tf.nn.gelu))
            fnn_layers.append(BatchNormalization())

        return keras.Sequential(fnn_layers, name=name)

    def train(self, x_train, y_train, x_valid, y_valid, *args, **kwargs):
        raise NotImplementedError()

    def predict(self, x_test):
        raise NotImplementedError()


class MLPModel(KerasModel):
    def train(self, x_train, y_train, x_valid, y_valid, *args, **kwargs):
        dense_inputs = keras.Input(shape=(x_train.shape[1],), name="raw_inputs")
        x = self.feedforward_layer(dense_inputs)
        outputs = self.output_layer(x)

        nn_model = keras.Model(inputs=dense_inputs, outputs=outputs, name="mlp_raw")
        print(nn_model.summary())

        nn_model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)

        nn_model.fit(
            x=x_train,
            y=y_train,
            validation_data=(x_valid, y_valid),
            epochs=self.hyperparams["epochs"],
            batch_size=self.hyperparams["batch_size"],
            callbacks=[self.callback_early_stopping, self.callback_decrease_lr]
        )

        self.model = nn_model

    def predict(self, x_test):
        return np.squeeze(self.model.predict(x_test))


class LogisticRegressionTrainer(KerasModel):
    def train(self, x_train, y_train, x_valid, y_valid, *args, **kwargs):
        dense_inputs = keras.Input(shape=(x_train.shape[1],), name="ohe_inputs")
        outputs = self.output_layer(dense_inputs)

        nn_model = keras.Model(inputs=dense_inputs, outputs=outputs, name="logistic_regression")
        print(nn_model.summary())

        nn_model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)

        # TODO one hot encode inputs

        nn_model.fit(
            x=x_train,
            y=y_train,
            validation_data=(x_valid, y_valid),
            epochs=self.hyperparams["epochs"],
            batch_size=self.hyperparams["batch_size"],
            callbacks=[self.callback_early_stopping, self.callback_decrease_lr]
        )
        return nn_model

    def predict(self, x_test):
        return np.squeeze(self.model.predict(x_test))


class EmbeddedBinModel(KerasModel):
    def __init__(self, numeric_features, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.numeric_features = numeric_features

    def create_inputs_and_embeddings(self, discrete_bin_vocab_size: int):
        inputs = []
        input_features = []

        for cf in self.numeric_features:
            cf_input = keras.Input(shape=(1,), name=f"{cf}_discrete")
            cf_feature = Embedding(discrete_bin_vocab_size + 1, 100, name=f"{cf}_embedding")(cf_input)

            inputs.append(cf_input)
            input_features.append(cf_feature)
        return inputs, input_features

    def train(self, x_train, y_train, x_valid, y_valid, *args, **kwargs):
        inputs, input_features = self.create_inputs_and_embeddings(kwargs["discrete_bin_vocab_size"])

        all_embeddings = tf.concat(input_features, axis=1)
        all_embeddings = Flatten()(all_embeddings)
        x = self.feedforward_layer(all_embeddings)
        outputs = self.output_layer(x)
        nn_model = keras.Model(inputs=inputs, outputs=outputs, name="quantised_bin_embeddings")
        print(nn_model.summary())

        nn_model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        model_inputs = {f"{cf}_discrete": x_train[cf] for i, cf in
                        zip(range(len(self.numeric_features)), self.numeric_features)}
        valid_inputs = {f"{cf}_discrete": x_valid[cf] for i, cf in
                        zip(range(len(self.numeric_features)), self.numeric_features)}

        nn_model.fit(
            x=model_inputs,
            y=y_train,
            validation_data=(valid_inputs, y_valid),
            epochs=self.hyperparams["epochs"],
            batch_size=self.hyperparams["batch_size"],
            callbacks=[self.callback_early_stopping, self.callback_decrease_lr]
        )

        self.model = nn_model

    def predict(self, x_test):
        test_inputs = {f"{cf}_discrete": x_test[cf] for i, cf in
                       zip(range(len(self.numeric_features)), self.numeric_features)}
        pred = np.squeeze(self.model.predict(test_inputs))
        return pred


class EmbeddedH3Model(KerasModel):
    def __init__(self, h3_resolutions: list, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.h3_resolutions = h3_resolutions

    def create_inputs_and_embeddings(self, embedding_vocab_size):
        h3_token_inputs = []
        h3_token_features = []

        for point in ["src", "dst"]:
            for h3_res in self.h3_resolutions:
                token_inputs = keras.Input(shape=(1,), name=f"spatial_tokens_h3_{point}_{h3_res}")
                token_features = Embedding(embedding_vocab_size[point][h3_res] + 1, 100,
                                           name=f"h3_embedding_{point}_{h3_res}")(token_inputs)

                h3_token_inputs.append(token_inputs)
                h3_token_features.append(token_features)

        return h3_token_inputs, h3_token_features

    def create_data_for_model(self, x: pd.DataFrame):
        train_inputs = {}
        for point in ["src", "dst"]:
            train_inputs_point = {f"spatial_tokens_h3_{point}_{k}": x[f"h3_hash_index_{point}_{k}"] for k in
                                  self.h3_resolutions}
            train_inputs = {**train_inputs, **train_inputs_point}
        return train_inputs

    def train(self, x_train, y_train, x_valid, y_valid, *args, **kwargs):

        all_token_inputs, all_token_features = self.create_inputs_and_embeddings(kwargs["embedding_vocab_size"])
        all_embeddings = tf.concat(all_token_features, axis=1)
        all_embeddings = Flatten()(all_embeddings)

        x = self.feedforward_layer(all_embeddings)
        outputs = self.output_layer(x)

        nn_model = keras.Model(inputs=all_token_inputs, outputs=outputs, name="h3_embedding_model")
        print(nn_model.summary())

        nn_model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)

        training_feature_inputs = []
        for dataset in [x_train, x_valid]:
            training_feature_inputs.append(self.create_data_for_model(dataset))

        nn_model.fit(
            x=training_feature_inputs[0],
            y=y_train,
            validation_data=(training_feature_inputs[1], y_valid),
            epochs=self.hyperparams["epochs"],
            batch_size=self.hyperparams["batch_size"],
            callbacks=[self.callback_early_stopping, self.callback_decrease_lr]
        )

        self.model = nn_model

    def predict(self, x_test):
        test_inputs = self.create_data_for_model(x_test)
        pred = np.squeeze(self.model.predict(test_inputs))
        return pred
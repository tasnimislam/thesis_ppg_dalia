from matplotlib import pyplot as plt
import numpy as np
from matplotlib import pyplot as plt
from IPython.display import clear_output
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Input,Convolution1D,BatchNormalization,Bidirectional,CuDNNLSTM,Dense,GlobalAveragePooling1D
from sklearn.metrics import mean_absolute_error
import tensorflow_privacy
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy

from privacy.custom_files.data_preprocessing import preprocess_according_paper2

# Global Variables

ECG_SAMPLING_RATE = 700
PPG_SAMPLING_RATE = 64
ACTIVITY_SAMPLING_RATE = 4
EMG_SAMPLING_RATE = 700

class PlotLearning(keras.callbacks.Callback):
    """
    Callback to plot the learning curves of the model during training.
    """

    def on_train_begin(self, logs={}):
        self.metrics = {}
        for metric in logs:
            self.metrics[metric] = []

    def on_epoch_end(self, epoch, logs={}):
        # Storing metrics
        for metric in logs:
            if metric in self.metrics:
                self.metrics[metric].append(logs.get(metric))
            else:
                self.metrics[metric] = [logs.get(metric)]

        # Plotting
        metrics = [x for x in logs if 'val' not in x]
        f, axs = plt.subplots(1, len(metrics), figsize=(15, 5))
        clear_output(wait=True)

        for i, metric in enumerate(metrics):
            axs[i].plot(range(1, epoch + 2),
                        self.metrics[metric],
                        label=metric)
            if logs['val_' + metric]:
                axs[i].plot(range(1, epoch + 2),
                            self.metrics['val_' + metric],
                            label='val_' + metric)

            axs[i].legend()
            axs[i].grid()

        plt.tight_layout()
        plt.savefig('o_p_new.png')
        plt.show()
        plt.close()



def CnnTransformerModel():
    i = Input(shape=(8 * PPG_SAMPLING_RATE, 1))

    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                           gamma_initializer='ones', moving_mean_initializer='zeros',
                           moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                           beta_constraint=None, gamma_constraint=None)(i)

    x = Convolution1D(8, kernel_size=10, strides=10, activation='relu', padding='same')(x)

    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                           gamma_initializer='ones', moving_mean_initializer='zeros',
                           moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
                           beta_constraint=None, gamma_constraint=None)(x)

    x = Bidirectional(CuDNNLSTM(128, return_sequences=True, return_state=False))(x)

    x = Bidirectional(CuDNNLSTM(64, return_sequences=True, return_state=False))(x)

    avg_pool = GlobalAveragePooling1D()(x)

    avg_pool = Dense(60, activation='relu')(avg_pool)

    y = Dense(1, activation='relu')(avg_pool)

    return Model(inputs=[i], outputs=[y])

def model_compile(mode, noise_multiplier, l2_norm_clip, num_microbatches, learning_rate, batch_size):
    model = CnnTransformerModel()

    if mode == 'general':
        model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.RMSprop(
            learning_rate=0.001,
            rho=0.9,
            momentum=0.0,
            epsilon=1e-07,
            centered=False,
            name="RMSprop"), metrics=['mean_absolute_error'])

        # model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.experimental.SGD(0.001, momentum=0.9), metrics=['mean_absolute_error'])

    elif mode == 'privacy':
        if batch_size % num_microbatches != 0:
            raise ValueError('Batch size should be an integer multiple of the number of microbatches')

        optimizer = tensorflow_privacy.DPKerasSGDOptimizer(
            l2_norm_clip=l2_norm_clip,
            noise_multiplier=noise_multiplier,
            num_microbatches=num_microbatches,
            learning_rate=learning_rate)

        loss = tf.keras.losses.MeanSquaredError()

        model.compile(optimizer=optimizer, loss=loss, metrics=['mean_absolute_error'])


    return model


def train(subject_no, epochs, l2_norm_clip, noise_multiplier, num_microbatches, learning_rate, path, mode = 'privacy'):
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = preprocess_according_paper2(subject_no, path)

    nb_epochs = epochs
    batch_size = 256
    weight_save_filename = f"weight_ECG_bilstm_subjectwise_{subject_no}.h5"

    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            print(model.trainable_variables)

    mdlcheckpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        weight_save_filename)
    callbacks_list = [mdlcheckpoint_cb]


    model = model_compile(mode, noise_multiplier, l2_norm_clip, num_microbatches, learning_rate, batch_size)
    model.fit(np.expand_dims(X_train, axis=2),
              y_train,
              epochs=nb_epochs,
              batch_size=batch_size,
              validation_data=(np.expand_dims(X_valid, axis=2), y_valid),
              verbose=1,
              shuffle=True,
              callbacks=callbacks_list
              )
    overall_epsilon = compute_dp_sgd_privacy.compute_dp_sgd_privacy(n=X_train.shape[0],
                                                      batch_size=batch_size,
                                                      noise_multiplier=noise_multiplier,
                                                      epochs=nb_epochs,
                                                      delta=1e-5)

    y_pred = model.predict(np.expand_dims(X_test, axis=2), batch_size=batch_size)
    return mean_absolute_error(y_test, y_pred), overall_epsilon
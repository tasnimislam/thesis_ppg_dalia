from keras.models import Model,Sequential
from keras import optimizers
from keras.layers import Input,Conv1D,BatchNormalization,MaxPooling1D,LSTM,Dense,Activation,Layer, Flatten
import keras.backend as K
import argparse
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from tensorflow import keras
from matplotlib import pyplot as plt
from IPython.display import clear_output
import tensorflow as tf
from ppg_data_preprocessing import preprocess_according_paper2

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
        f, axs = plt.subplots(1, len(metrics), figsize=(15,5))
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
        plt.show()
        
def CnnTransformerModel():
    i = Input(shape = (8*PPG_SAMPLING_RATE, 1))
    
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)(i)

    x = Convolution1D(8, kernel_size = 10, strides = 10, activation='relu', padding = 'same')(x)
    
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)(x)
    
    x = Bidirectional(CuDNNLSTM(128, return_sequences = True, return_state = False))(x)
    
    x = Bidirectional(CuDNNLSTM(64, return_sequences = True, return_state = False))(x)
    
    avg_pool = GlobalAveragePooling1D()(x)
    
    avg_pool = Dense(60,activation = 'relu')(avg_pool)
    
    y = Dense(1,activation = 'relu')(avg_pool)
    
    return Model(inputs = [i], outputs = [y])

def model_compile():
    model = CnnTransformerModel()
    
    model.compile(loss='mean_squared_error', optimizer= tf.keras.optimizers.RMSprop(
        learning_rate=0.001,
        rho=0.9,
        momentum=0.0,
        epsilon=1e-07,
        centered=False,
        name="RMSprop"),metrics = ['mean_absolute_error'])
    return model

def train(subject_no):
    
    model = model_compile()

    model.compile(loss='mean_squared_error', optimizer= tf.keras.optimizers.RMSprop(
        learning_rate=0.001,
        rho=0.9,
        momentum=0.0,
        epsilon=1e-07,
        centered=False,
        name="RMSprop"),metrics = ['mean_absolute_error'])
   
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = preprocess_according_paper2(subject_no)
    
    nb_epochs = 100
    batch_size = 256
    weight_save_filename = "weight_ECG_bilstm_subjectwise.h5"

    mdlcheckpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        weight_save_filename)
    callbacks_list = [PlotLearning(), mdlcheckpoint_cb]
    model.fit(np.expand_dims(X_train, axis = 2),
              y_train,
              epochs = nb_epochs,
              batch_size = batch_size,
              validation_data=(np.expand_dims(X_valid, axis = 2), y_valid),
              verbose=1,
              shuffle=True,
              callbacks=callbacks_list
              )
    y_pred = model.predict(np.expand_dims(X_test, axis = 2), batch_size = 256)
    return mean_absolute_error(y_test, y_pred)

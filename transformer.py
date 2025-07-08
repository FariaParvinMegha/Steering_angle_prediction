import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.models import Model,Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Dense, Dropout, Flatten , Reshape
from keras.layers import MultiHeadAttention, LayerNormalization, Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from utils import INPUT_SHAPE, batch_generator
import argparse
import os
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

np.random.seed(0)

def load_data(args):
    # Load data as before
    data_df = pd.read_csv(os.path.join(os.getcwd(), args.data_dir, 'data_log.csv'),
                          names=['center', 'left', 'right', 'steering'])
    X = data_df[['center', 'left', 'right']].values
    y = data_df['steering'].values
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=args.test_size, random_state=0)
    return X_train, X_valid, y_train, y_valid

def build_transformer_model(input_shape, args):
    # Input layer
    input_layer = Input(shape=input_shape)
    
    # Convolutional layers for feature extraction
    x = Lambda(lambda x: x / 127.5 - 1.0)(input_layer)
    x = Conv2D(24, (5, 5), activation='elu', strides=(2, 2))(x)
    x = Conv2D(36, (5, 5), activation='elu', strides=(2, 2))(x)
    x = Conv2D(48, (5, 5), activation='elu', strides=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation='elu')(x)
    x = Conv2D(64, (3, 3), activation='elu')(x)
    x = Dropout(args.keep_prob)(x)
    x = Flatten()(x)
    

    x = Reshape((-1, 64))(x)  
    
    # Transformer layer
    transformer_output = MultiHeadAttention(num_heads=8, key_dim=64)(x, x)
    

    y = Flatten()(transformer_output)  # Flatten the transformer output
    y = Dense(100, kernel_initializer='random_uniform', bias_initializer='zeros', activation='elu')(y)
    y = Dense(50, activation='elu')(y)
    y = Dense(10, activation='elu')(y)
    output_layer = Dense(1)(y)

    # Model
    model = Model(inputs=input_layer, outputs=output_layer)
    model.summary()
    
    return model


def s2b(s):
    s = s.lower()
    return s == 'true' or s == 'yes' or s == 'y' or s == '1'


def train_model(model, args, X_train, X_valid, y_train, y_valid):
    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=args.save_best_only,
                                 mode='auto')

    model.compile(loss='mean_squared_error', optimizer=Adam(lr=args.learning_rate))

    model.fit(batch_generator(args.data_dir, X_train, y_train, args.batch_size, True),
              epochs=args.nb_epoch,
              batch_size=args.batch_size,
              validation_data=batch_generator(args.data_dir, X_valid, y_valid, args.batch_size, False),
              callbacks=[checkpoint],
              verbose=1)

def main():
    parser = argparse.ArgumentParser(description='Behavioral Cloning')
    parser.add_argument('-d', help='data directory',        dest='data_dir',          type=str,   default='')
    parser.add_argument('-t', help='test size fraction',    dest='test_size',         type=float, default=0.2)
    parser.add_argument('-k', help='drop out probability',  dest='keep_prob',         type=float, default=0.5)
    parser.add_argument('-n', help='number of epochs',      dest='nb_epoch',          type=int,   default=10)
    parser.add_argument('-s', help='samples per epoch',     dest='samples_per_epoch', type=int,   default=4000)
    parser.add_argument('-b', help='batch size',            dest='batch_size',        type=int,   default=2)
    parser.add_argument('-o', help='save best models only', dest='save_best_only',    type=s2b,   default='true')
    parser.add_argument('-l', help='learning rate',         dest='learning_rate',     type=float, default=1.0e-4)
    args = parser.parse_args()

    print('-' * 30)
    print('Parameters')
    print('-' * 30)
    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)


    data = load_data(args)
    input_shape = (66, 200, 3)  
    model = build_transformer_model(input_shape,args)
    train_model(model, args, *data)

if __name__ == '__main__':
    main()

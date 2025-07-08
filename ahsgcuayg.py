import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.models import Sequential,Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten,Reshape,Input,MultiHeadAttention,Reshape, UpSampling2D, Concatenate ,Resizing,BatchNormalization
from utils import INPUT_SHAPE, batch_generator
from keras.applications import VGG16,VGG19,MobileNetV2,EfficientNetB2,ResNet50,InceptionResNetV2,InceptionV3
from vit_keras import vit
import argparse
import os

np.random.seed(0)


def load_data(args):
    data_df = pd.read_csv(os.path.join(os.getcwd(), args.data_dir, 'data_log.csv'),
                          names=['center', 'left', 'right', 'steering'])
    X = data_df[['center', 'left', 'right']].values
    y = data_df['steering'].values
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=args.test_size, random_state=0)
    return X_train, X_valid, y_train, y_valid



def build_model(args):
    # Input layer
    input_layer = Input(shape=INPUT_SHAPE)
    

    # vgg16_model = MobileNetV2(include_top = False,
    #                      weights = 'imagenet', 
    #                      input_tensor = input_layer)
    # vgg16_model.trainable = True
    vitmodel = vit.vit_b16(
        image_size = (66,200),
        activation = 'softmax',
        pretrained = True,
        include_top = False,
        pretrained_top = False,
        classes = 5)
    model=Sequential()
    model.add(vitmodel)
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(100, kernel_initializer='he_normal',
                    bias_initializer='zeros', activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))



    # Model
    # model.summary()
    
    return model


def train_model(model, args, X_train, X_valid, y_train, y_valid):
    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=args.save_best_only,
                                 mode='auto')

    model.compile(loss='mean_squared_error', optimizer=Adam(lr=args.learning_rate))

    model.fit(batch_generator(args.data_dir, X_train, y_train, args.batch_size, True),
              steps_per_epoch=args.samples_per_epoch,
              epochs=args.nb_epoch,
              validation_data=batch_generator(args.data_dir, X_valid, y_valid, args.batch_size, False),
              validation_steps=len(X_valid) // args.batch_size,
              callbacks=[checkpoint],
              verbose=1)


def s2b(s):
    s = s.lower()
    return s == 'true' or s == 'yes' or s == 'y' or s == '1'


def main():
    parser = argparse.ArgumentParser(description='Behavioral Cloning')
    parser.add_argument('-d', help='data directory',        dest='data_dir',          type=str,   default='')
    parser.add_argument('-t', help='test size fraction',    dest='test_size',         type=float, default=0.2)
    parser.add_argument('-k', help='drop out probability',  dest='keep_prob',         type=float, default=0.5)
    parser.add_argument('-n', help='number of epochs',      dest='nb_epoch',          type=int,   default=10)
    parser.add_argument('-s', help='samples per epoch',     dest='samples_per_epoch', type=int,   default=10000)
    parser.add_argument('-b', help='batch size',            dest='batch_size',        type=int,   default=4)
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
    model = build_model(args)
    train_model(model, args, *data)


if __name__ == '__main__':
    main()

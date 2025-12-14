from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input


def build_ann_rgb(input_shape, num_classes=3):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Flatten())
    model.add(Dense(20, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(5, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(num_classes, activation='softmax'))
    return model


def build_ann_grayscale(input_shape, num_classes=3):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Flatten())
    model.add(Dense(50, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(20, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(5, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(num_classes, activation='softmax'))
    return model


def build_ann_blur(input_shape, num_classes=3):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Flatten())
    model.add(Dense(50, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(20, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(5, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(num_classes, activation='softmax'))
    return model


def build_ann_laplacian(input_shape, num_classes=3):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Flatten())
    model.add(Dense(50, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(20, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(5, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(num_classes, activation='softmax'))
    return model

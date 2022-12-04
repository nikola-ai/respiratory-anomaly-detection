import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, Dense, Dropout, MaxPool2D, Flatten, \
    LeakyReLU


def get_kernel_model(input_height, input_width, learning_rate):
    K.clear_session()

    model = Sequential()
    model.add(
        Conv2D(128, [7, 11], strides=[2, 2], padding='SAME',
               input_shape=(input_height, input_width, 1)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPool2D(padding='SAME'))

    model.add(Conv2D(256, [5, 5], padding='SAME'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPool2D(padding='SAME'))

    model.add(Conv2D(256, [1, 1], padding='SAME'))
    model.add(Conv2D(256, [3, 3], padding='SAME'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPool2D(padding='SAME'))

    model.add(Conv2D(512, [1, 1], padding='SAME'))
    model.add(Conv2D(512, [3, 3], padding='SAME', activation='relu'))
    model.add(Conv2D(512, [1, 1], padding='SAME'))
    model.add(Conv2D(512, [3, 3], padding='SAME', activation='relu'))
    model.add(MaxPool2D(padding='SAME'))
    model.add(Flatten())

    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(512, activation='relu'))
    model.add(Dense(4, activation='softmax'))

    opt = tf.keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999,
                                   decay=0.00, amsgrad=False)
    opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)

    model.compile(optimizer=opt, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

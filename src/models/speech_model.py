import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, Dense, Dropout, MaxPool2D, \
    GlobalMaxPool2D, LeakyReLU, BatchNormalization


def get_speech_model(input_height, input_width, learning_rate):
    K.clear_session()

    model = Sequential()
    model.add(Conv2D(64, [7, 3], input_shape=(input_height, input_width, 1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(MaxPool2D(padding='SAME', pool_size=[1, 3]))

    model.add(Conv2D(128, [1, 7]))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(MaxPool2D(padding='SAME', pool_size=[1, 4]))

    model.add(Conv2D(256, [1, 10], padding='VALID'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(512, [7, 1]))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(GlobalMaxPool2D())

    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))

    opt = tf.keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999,
                                   decay=0.00, amsgrad=False)
    opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)

    model.compile(optimizer=opt, loss='categorical_crossentropy',
                  metrics=None)  # Metrics are calculated in callbacks
    return model

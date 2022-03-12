import tensorflow as tf

def reshape(input_data, label_data):
    input_data = input_data.reshape(input_data.shape[0], 1,
                                    input_data.shape[1] * input_data.shape[2])
    input_data = input_data.astype('float32')
    input_data /= 255
    # encode output which is a number in range [0,9] into a vector of size 10
    # e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    label_data = tf.keras.utils.to_categorical(label_data)
    return input_data, label_data

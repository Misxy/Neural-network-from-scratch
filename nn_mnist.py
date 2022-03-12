from ast import arg
from importers import *

def main():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, y_train = utils.reshape(x_train, y_train)
    x_test, y_test = utils.reshape(x_test, y_test)

    # Create our network
    net = Network()
    # input_shape=(1, 28*28) output_shape=(1, 100)
    net.add(FCLayer(x_train.shape[2], 100))
    net.add(ActivationLayer(tanh, tanh_prime))
    # input_shape=(1, 100) output_shape=(1, 50)
    net.add(FCLayer(100, 50))
    net.add(ActivationLayer(tanh, tanh_prime))
    # input_shape=(1, 50) output_shape=(1, 10)
    net.add(FCLayer(50, y_test.shape[1]))
    net.add(ActivationLayer(tanh, tanh_prime))

    net.loss_fn(mse, mse_prime)
    net.fit(x_train[:100], y_train[:100], epochs=args.epoch, learning_rate=args.learning_rate)

    # test on 5 samples
    first_index = 0
    last_index = 5
    outputs = net.predict(x_test[first_index:last_index])
    print("Predicted values : {}".format(outputs))
    print("Actual values : {}".format(y_test[first_index:last_index]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="List of parameters")
    parser.add_argument('-e',
                        '--epoch',
                        action='store',
                        type=int,
                        help='A Number of epoch hyper-parameter')
    parser.add_argument('-lr',
                        '--learning_rate',
                        action='store',
                        type=float,
                        help='A learning rate hyper-parameter.')
    args = parser.parse_args()
    main()

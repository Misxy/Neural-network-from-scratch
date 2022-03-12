from importers import *

# training data
x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

# network
net = Network()
net.add(FCLayer(2, 3))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(3, 1))
net.add(ActivationLayer(tanh, tanh_prime))

# train
net.loss_fn(mse, mse_prime)
net.fit(x_train, y_train, epochs=10000, learning_rate=0.1)

# test
out = net.predict(x_train)
print('Predicted output is: {0}'.format(out))
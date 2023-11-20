# -*- coding: utf-8 -*-
import numpy as np
import struct

import matplotlib.pyplot as plt

def readMNISTdata():
    with open('t10k-images-idx3-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        test_data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        test_data = test_data.reshape((size, nrows*ncols))

    with open('t10k-labels-idx1-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        test_labels = np.fromfile(
            f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        test_labels = test_labels.reshape((size, 1))

    with open('train-images-idx3-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        train_data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        train_data = train_data.reshape((size, nrows*ncols))

    with open('train-labels-idx1-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        train_labels = np.fromfile(
            f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        train_labels = train_labels.reshape((size, 1))

    # augmenting a constant feature of 1 (absorbing the bias term)
    train_data = np.concatenate(
        (np.ones([train_data.shape[0], 1]), train_data), axis=1)
    test_data = np.concatenate(
        (np.ones([test_data.shape[0], 1]),  test_data), axis=1)
    _random_indices = np.arange(len(train_data))
    np.random.shuffle(_random_indices)
    train_labels = train_labels[_random_indices]
    train_data = train_data[_random_indices]

    X_train = train_data[:50000] / 256
    t_train = train_labels[:50000]

    X_val = train_data[50000:] / 256
    t_val = train_labels[50000:]

    return X_train, t_train, X_val, t_val, test_data / 256, test_labels


def cross_entropy_loss(y, t):
    N = len(t)
    loss = 0
    for rowNum in range(t.shape[0]):
        item_t = t[rowNum][0]
        item_y = y[rowNum][item_t]
        loss -=np.log(item_y)
    return loss / N

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def train(X_train, t_train, X_val, t_val, learning_rate, epochs=1000, batch_size=50):

    num_samples, num_features = X_train.shape
    num_classes = len(np.unique(t_train))

    # Initialize weights (including bias)
    W = np.random.randn(num_features, num_classes)
    lossBest = 1000
    train_losses = []
    accuracy_batches = []
    valid_accs = []
    # Gradient Descent
    for epoch in range(epochs):
        # Forward pass
        X_batch = X_train[epoch: epoch + batch_size]
        t_batch = t_train[epoch: epoch + batch_size]
        y = softmax(X_batch @ W)
        
        # Compute gradient of the loss with respect to weights
        gradient = X_batch.T @ (y - (np.eye(num_classes)[t_batch.flatten()]))

        # Update weights
        W -= learning_rate * gradient / batch_size

        # Compute and print loss
        loss_batch = cross_entropy_loss(y, t_batch)
        train_losses.append(loss_batch)
        accuracy_batches.append(predict(X_batch, W, t_batch)[-1])
        valid_accs.append(predict(X_val, W, t_val)[-1])
        print("valid_accs: ",valid_accs[-1], "accuracy_batches: ", accuracy_batches[-1])

        if loss_batch<=lossBest:
            epoch_best = epoch+1
            W_best = W
            acc_best = accuracy_batches[-1]
    return epoch_best, acc_best,  W_best, train_losses, valid_accs


def predict(X, W, t=None):
    # X_new: Nsample x (d+1)
    # W: (d+1) x K
    y = softmax(X @ W)
    t_hat = np.array(np.argmax(y, axis=1)).reshape(len(np.argmax(y, axis=1)),1)
    loss = cross_entropy_loss(y, t)
    acc = np.sum(t_hat == t)/len(t)
    return y, t_hat, loss, acc

# Main code starts here
X_train, t_train, X_val, t_val, X_test, t_test = readMNISTdata()


print(X_train.shape, t_train.shape, X_val.shape,
      t_val.shape, X_test.shape, t_test.shape)


N_class = 10

alpha = 0.1     # learning rate
batch_size = 50    # batch size
MaxEpoch = 100        # Maximum epoch
decay = 0.          # weight decay


# TODO: report 3 number, plot 2 curves
epoch_best, acc_best,  W_best, train_losses, valid_accs = train(X_train, t_train, X_val, t_val, alpha, MaxEpoch , batch_size)

acc_test = predict(X_test, W_best, t_test)[-1]

print(f'number of epoch that yields the best validation performance: {epoch_best}')
print(f'validation performance (accuracy) in that epoch:{valid_accs[epoch_best-1]}')
print(f' test performance (accuracy) in that epoch: {acc_test}')

# ploting the figures

plt.plot(range(1, len(train_losses)+1), train_losses, label='Training Loss')
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.title('Train Loss')
plt.legend()
plt.savefig('train_loss.png')
plt.show()

plt.plot(range(1, len(valid_accs)+1), valid_accs, label='Validation Accuracy')
plt.xlabel('epochs')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy')
plt.legend()
plt.savefig('valid_acc.png')
plt.show()

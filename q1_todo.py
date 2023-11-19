from utils import plot_data, generate_data
import numpy as np


def sigmoid(X, w, b):
    b_temp = np.array(X.shape[0]*[b[0]])
    z = X @ w + b_temp
    return 1 / (1 + np.exp(-z))


def train_logistic_regression(X, t):
    """
    Given data, train your logistic classifier.
    Return weight and bias
    """
    # initial w,b
    w = np.zeros(X.shape[1]).T
    b = np.zeros(X.shape[1]).T


    def grad(x,w,b,t):
        z = sigmoid(x,w,b)
        grad_b = (z-t)
        grad_w = (z-t)*x.T
        return grad_w,grad_b

    error = 1000
    epsilon = 0.3
    alpha = 0.1


    while error >= epsilon:
        for row_counter in range(X.shape[0]):
            x = X[row_counter,:]
            grad_w,grad_b = grad(x,w,b,t[row_counter])
            
            # update the parameters
            w = w - alpha * grad_w
            b = b - alpha * grad_b

        # Calculate the error
        error = (1/X.shape[0])*np.linalg.norm(predict_logistic_regression(X, w, b)-t)
    return w, b


def predict_logistic_regression(X, w, b):
    """
    Generate predictions by your logistic classifier.
    """
    # Evaluate Sigmoid
    sigmoid_X = sigmoid(X,w,b)
    t = (sigmoid_X>=0.5).astype(int)
    return t


def train_linear_regression(X, t):
    """
    Given data, train your linear regression classifier.
    Return weight and bias
    """
    X = np.concatenate((X, np.ones((X.shape[0],1))), axis=1)
    results = np.linalg.inv(X.T @ X) @ X.T @ t
    w = results[:-1]
    b = results[-1]
    return w, b


def predict_linear_regression(X, w, b):
    """
    Generate predictions by your logistic classifier.
    """
    # apply linear transform
    temp = X@w+b
    t = (temp>=0).astype(int)
    return t


def get_accuracy(t, t_hat):
    """
    Calculate accuracy,
    """
    return (1/t.shape[0])*(np.sum(t==t_hat))*100


def main():
    # Dataset A
    # Linear regression classifier
    X, t = generate_data("A")
    w, b = train_linear_regression(X, t)
    t_hat = predict_linear_regression(X, w, b)
    print("Accuracy of linear regression on dataset A:", get_accuracy(t_hat, t))
    plot_data(X, t, w, b, is_logistic=False,
              figure_name='dataset_A_linear.png')

    # logistic regression classifier
    X, t = generate_data("A")
    w, b = train_logistic_regression(X, t)
    t_hat = predict_logistic_regression(X, w, b)
    print("Accuracy of logistic regression on dataset A:", get_accuracy(t_hat, t))
    plot_data(X, t, w, b, is_logistic=True,
              figure_name='dataset_A_logistic.png')

    # Dataset B
    # Linear regression classifier
    X, t = generate_data("B")
    w, b = train_linear_regression(X, t)
    t_hat = predict_linear_regression(X, w, b)
    print("Accuracy of linear regression on dataset B:", get_accuracy(t_hat, t))
    plot_data(X, t, w, b, is_logistic=False,
              figure_name='dataset_B_linear.png')

    # logistic regression classifier
    X, t = generate_data("B")
    w, b = train_logistic_regression(X, t)
    t_hat = predict_logistic_regression(X, w, b)
    print("Accuracy of logistic regression on dataset B:", get_accuracy(t_hat, t))
    plot_data(X, t, w, b, is_logistic=True,
              figure_name='dataset_B_logistic.png')


main()

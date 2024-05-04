import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt

data  = pd.read_csv('train.csv')

data = np.array(data)
m, n = data.shape
np.random.shuffle(data) 

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape


def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A
    
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def ReLU_deriv(Z):
    return Z > 0

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2


def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()


def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')


def save_parameters(W1, b1, W2, b2, filename="parameters.npy"):
    np.save(filename, {"W1": W1, "b1": b1, "W2": W2, "b2": b2})
    print(f"Parameters saved to {filename}")

def load_parameters(filename="parameters.npy"):
    try:
        params = np.load(filename, allow_pickle=True).item()
        W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]
        print(f"Parameters loaded from {filename}")
        return W1, b1, W2, b2
    except FileNotFoundError:
        print(f"File {filename} not found. Returning random parameters.")
        return init_params()


def main():
    while True:
        print("||Resistance Optimization for Neural networks using PID||")
        print("t. Train Network")
        print("s. show prediction on a sample of data")
        print("a. Get accuracy of the network")
        print("l. load parameters")
        print("q. Quit")
        choice = input("Input: ")
        if choice == 'a':
            clear_terminal()
            dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
            get_accuracy(dev_predictions, Y_dev)
            input("Press any key to continue. ")
            clear_terminal()
        elif choice =='t':
            itr = int(input("set iterations: "))
            W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, itr)
            print()
        elif choice =='s':
            clear_terminal()
            test_prediction(np.random.randint(0, m_train), W1, b1, W2, b2)
            print()
        elif choice =='l':
            clear_terminal()
            W1, b1, W2, b2 = load_parameters(filename="parameters.npy")
            print()
        elif choice =='q':
            save_choice = input("Do you want to save the current parameters? (y/n): ").lower()
            if save_choice == 'y':
                save_parameters(W1, b1, W2, b2)
            break
    
if __name__ == "__main__":
   main()
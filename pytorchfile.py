import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as tts
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import random
from pid import PIDOptimizer
import torch.optim as optim
import os

# Parameters
input_size = 784
hidden_size = 256
num_classes = 10
num_epochs = 20
batch_size = 100
learning_rate = 0.01
random.seed(20)

I=3
I = float(I)
D = 100
D = float(D)

losses = []
epoch_losses = []

# Data Loading
my_df = pd.read_csv('train.csv')

x = my_df.drop('label', axis=1)
y = my_df['label']
x = x.values
y = y.values

x_train, x_test, y_train, y_test = tts(x, y, test_size=0.3)
x_train = torch.cuda.FloatTensor(x_train) / 255.0
x_test = torch.FloatTensor(x_test) / 255.0
y_train = torch.cuda.LongTensor(y_train)
y_test = torch.LongTensor(y_test)
x_train = x_train.cuda()
y_train = y_train.cuda()

train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Model Initialization
class TheModelClass(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        #self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        #out = self.fc2(out)
        #out = F.relu(out)
        out = self.fc3(out)
        #out = F.softmax(out)
        return out

# Setting the Model
model = TheModelClass(input_size, hidden_size, num_classes)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss().cuda()

#optimizer = torch.optim.Adam(model.parameters(), learning_rate)
#optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
optimizer = PIDOptimizer(model.parameters(), lr=learning_rate, weight_decay=0.0001, momentum=0.9, I=I, D=D)

# Set the Model to Train with CUDA
model.cuda()
model.train()

def train_model():
    clear_terminal()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            y_pred = model.forward(batch_x)
            loss = criterion(y_pred, batch_y)
            losses.append(loss.cpu().detach().numpy())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_losses.append(epoch_loss / len(train_loader))

        model.eval()
        with torch.no_grad():
            y_pred = model.forward(x_test.cuda())
            _, predicted = torch.max(y_pred, 1)
            correct = (predicted == y_test.cuda()).sum().item()
            total = y_test.size(0)
            accuracy = correct / total

            if epoch == 0:
                clear_terminal()
                continue
            elif epoch % 3 == 0:
                print(f'Step: {epoch}/{num_epochs} | loss : {epoch_loss / len(train_loader)} | Test Accuracy: {accuracy * 100:.2f}%')

def plot_losses():
    plt.plot(range(num_epochs), epoch_losses)
    plt.ylabel("Average loss per epoch")
    plt.xlabel('Epoch')
    plt.show()

def test_model():
    model.eval()
    with torch.no_grad():
        y_pred = model.forward(x_test.cuda())
        _, predicted = torch.max(y_pred, 1)
        correct = (predicted == y_test.cuda()).sum().item()
        total = y_test.size(0)
        accuracy = correct / total
        print(f'Test Accuracy: {accuracy * 100:.2f}%')


def show_random_prediction():
    model.eval()
    index = random.randint(0, len(x_test) - 1)
    data_point = x_test[index].unsqueeze(0)  # Add a batch dimension
    label = y_test[index].item()

    with torch.no_grad():
        y_pred = model.forward(data_point.cuda())
        _, predicted = torch.max(y_pred, 1)
        predicted_label = predicted.item()

    print(f"Model Prediction: {predicted_label}, True Label: {label}")

def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    global model
    while True:
        print("|| Resistance Optimization for Neural networks using PID ||")
        print("t. Train Model")
        print("p. Plot Losses")
        print("e. Test Model")
        print("r. Show Random Prediction")
        print("s. save the parameters")
        print("l. load the parameters")
        print("q. Quit")
        choice = input("Enter your choice: ")

        if choice == 't':
            clear_terminal()
            train_model()
            print("\n\n")
        elif choice == 'p':
            clear_terminal()
            plot_losses()
        elif choice == 'e':
            clear_terminal()
            test_model()
            print("\n\n")
        elif choice == 'r':
            clear_terminal()
            print("")
            show_random_prediction()
            print("\n\n")
        elif choice == 's':
            clear_terminal()
            param = "model_params.pth"
            torch.save(model.state_dict(), param)
            print("\n\n")
        elif choice == 'l':
            clear_terminal()
            param = "model_params.pth"
            model = TheModelClass(input_size, hidden_size, num_classes)  
            model.load_state_dict(torch.load(param))
            model.cuda()  
            model.eval()
        elif choice == 'q':
            break
        else:
            print("Invalid choice. Please enter a valid option.")

if __name__ == "__main__":
    main()

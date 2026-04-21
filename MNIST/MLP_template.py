from binascii import a2b_base64
from math import log
from turtle import forward
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt

def sigmoid(x):  # manually define the sigmoid
    return 1/(1 + np.exp(-x))

def softmax(x):  # define the softmax 
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

def dataloader(train_dataset, test_dataset, batch_size=128):
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def load_data(batch_size):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = torchvision.datasets.MNIST(root="./data/mnist", train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root="./data/mnist", train=False, download=True, transform=transform)
    print("The number of training data:", len(train_dataset))
    print("The number of testing data:", len(test_dataset))
    return dataloader(train_dataset, test_dataset, batch_size)

class MLP:
    def __init__(self, input_size, hidden_size, a, lr):  # 
        
        self.lr = lr
        
        self.Weight1 = np.random.randn(hidden_size, input_size) * 0.01
        self.bias1 = np.zeros((hidden_size, 1))
        self.Weight2 = np.random.randn(a, hidden_size) * 0.01
        self.bias2 = np.zeros((a, 1))    
        
        self.z1 = None
        self.a1 = None
        self.z2 = None
    
    def forward(self, x):  # forward propagation to get predictions
        # z = W * a + b
        # a = g(z)
        z1 = (x @ self.Weight1.T) + self.bias1.T # self.bias1 * 1^T: in lecture multiply by transposed 1 matrix
        a1 = sigmoid(z1)
        
        z2 = (a1 @ self.Weight2.T) + self.bias2.T
        outputs = softmax(z2) 
        
        self.z1 = z1
        self.a1 = a1 
        self.z2 = z2
        
        # 10 by 128 vector
        # rows of class guesses x 128 images per batch
        return outputs
    
    # pred = output from forward 
    # y = true labels
    # x = input (128 by 784 matrix)
    def backward(self, x, y, pred):
        batch_size = x.shape[0]
        # one-hot encode the labels

        # compute the gradients
        
        # sigmoid + softmax loss function derivation result in y^ - y
        # dZ2 is guess - true class
        dz2 = pred - y
        #updated weight of second layer
        #dW^(l) = dZ^(l) * A^(l-1)^T
        dw2 = np.dot(dz2.T, self.a1) / batch_size
        #updated bias of second layer
        db2 = np.dot(dz2.T, np.ones((batch_size, 1))) / batch_size 
        #dA^(l-1) = W^(l) * dZ^(l)
        dA1 = np.dot(dz2, self.Weight2)

        dz1 = (self.a1 * (1-self.a1)) * dA1
        dw1 = np.dot(dz1.T, x) / batch_size
        db1 = np.dot(dz1.T, np.ones((batch_size, 1))) / batch_size
        
        # update the weights and biases
        self.Weight2 -= self.lr * dw2
        self.bias2 -= self.lr *db2
        
        self.Weight1 -= self.lr * dw1
        self.bias1 -= self.lr * db1
        

    def train(self, x,y):
        # call forward function
        output = self.forward(x)
        # calculate loss,
        # 1e-12 avoid log(0)
        epsilon = 1e-12
        output = np.clip(output, epsilon, 1-epsilon)
        loss = np.mean(-np.sum(y*np.log(output), axis=1))
        # call backward function
        self.backward(x, y, output)
        
        return loss

def main():
    batch_size = 64
    # First, load data
    train_loader, test_loader = load_data(batch_size)

    # Second, define hyperparameters
    input_size = 28*28  # MNIST images are 28x28 pixels
    num_epochs = 100

    hidden_layer_size = 192
    num_classes = 10
    learning_rate = 0.099 

    # model object here
    model = MLP(input_size=input_size, hidden_size=hidden_layer_size, a=num_classes, lr=learning_rate)

    # Then, train the model
    start_time = time.time()
    epoch_losses = []
    for epoch in range(num_epochs):
        total_loss = 0

        for inputs, labels in train_loader:  # define training phase for training model
            batch_size = inputs.shape[0]
            x = inputs.view(batch_size, input_size).numpy()
            y_labels = labels.numpy()
            y = np.eye(num_classes)[y_labels]

            loss = model.train(x,y)
            total_loss += loss
        avg_epoch_loss = total_loss / len(train_loader)
        epoch_losses.append(avg_epoch_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss}") # print the loss for each epoch

    end_time = time.time()
    training_time = end_time - start_time
    print(f"Total training time: {training_time:.2f} seconds")
 
    # Finally, evaluate the model
    correct_pred = 0
    total_pred = 0

    eval_start_time = time.time()

    for inputs, labels in test_loader:
        x = inputs.view(-1, input_size).numpy()
        y = labels.numpy()
        pred = model.forward(x)  # the model refers to the model that was trained during the training phase
        predicted_labels = np.argmax(pred, 1)
        correct_pred += np.sum(predicted_labels == y)
        total_pred += len(labels)

    eval_end_time = time.time()
    eval_time = eval_end_time - eval_start_time
    print(f"Test Accuracy: {correct_pred/total_pred}")
    print(f"Total evaluation time: {eval_time:.2f} seconds")
    
    total_elapsed_time = training_time + eval_time
    print(f"Total time (training + evaluation): {total_elapsed_time:.2f} seconds")

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, num_epochs + 1), epoch_losses, marker="o")
    plt.title("MLP Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("mlp_training_loss.png")
    plt.show()


if __name__ == "__main__":  # Program entry
    main()  
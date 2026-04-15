from binascii import a2b_base64
from math import log
from turtle import forward
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def sigmoid(x):  # manually define the sigmoid
    return 1/(1 + np.exp(-x))

def softmax(x):  # define the softmax 
    return np.exp(x) / sum(np.exp(x), axis=1, keepdims=True)

def dataloader(train_dataset, test_dataset, batch_size=128):
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def load_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = torchvision.datasets.MNIST(root="./data/mnist", train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root="./data/mnist", train=False, download=True, transform=transform)
    print("The number of training data:", len(train_dataset))
    print("The number of testing data:", len(test_dataset))
    return dataloader(train_dataset, test_dataset)

class MLP:
    def __init__(self, input_size, hidden_size, a,lr):  # 
        
        self.lr = lr
        
        self.Weight1 = np.random.randn(hidden_size, input_size) * 0.01
        self.bias1 = np.zeros((hidden_size, 1))
        self.Weight2 = np.random.randn(a, hidden_size) * 0.01
        self.bias2 = np.zeros((a, 1))    
        
        self.z1 = None
        self.a1 = None
        self.z2 = None
        self.a2 = None
    
    def forward(self, x):  # forward propagation to get predictions
        # z = W * a + b
        # a = g(z)
        z1 = (self.Weight1 @ x) + self.bias1
        a1 = sigmoid(z1)
        
        z2 = (self.Weight2 @ a1) + self.bias2
        outputs = softmax(z2) 
        
        self.z1 = z1
        self.a1 = a1 
        self.z2 = z2
        self.a2 = outputs
        
        # 10 by 128 vector
        # rows of class guesses x 128 images per batch
        return outputs
    
    # pred = output from forward 
    # y = true labels
    # x = input
    def backward(self, x, y, pred):
        batch_size = x.shape[1]
        # one-hot encode the labels

        # compute the gradients
        
        # dZ2 is guess - true class
        dz2 = self.a2 - y
        #updated weight of second layer
        #dW^(l) = dZ^(l) * A^(l-1)^T
        dw2 = np.dot(dz2, self.a1.T) / batch_size
        #updated bias of second layer
        db2 = np.dot(dz2 * 1) / batch_size
        #dA^(l-1) = W^(l) * dZ^(l)
        dA1 = np.dot(self.Weight2.T * dz2) / batch_size
        
        dz1 = np.dot((self.a1 * (1-self.a1)), dA1)
        dw1 = np.dot(dz1, x.T) / batch_size
        db1 = sum(dz1, axis=1, keepdims=True) / batch_size
        
        # update the weights and biases
        self.Weight2 -= self.lr * dw2
        self.bias2 -= self.lr *db2
        
        self.Weight1 -= self.lr * dw1
        self.bias1 -= self.lr * db1
        

    def train(self, x,y):
        # call forward function
        output = self.forward(x)
        # calculate loss
        inner = np.sum(y, np.log(output + 1e-12), axis=1)
        loss = -np.mean(inner)
        # call backward function
        self.backward(x, y, output)
        
        return loss

def main():
    # First, load data
    train_loader, test_loader = load_data()

    # Second, define hyperparameters
    input_size = 28*28  # MNIST images are 28x28 pixels
    num_epochs = 100


    # Then, train the model
    for epoch in range(num_epochs):
        total_loss = 0

        for inputs, lables in train_loader:  # define training phase for training model
            

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader)}") # print the loss for each epoch
        
    num_neurons_hidden = 128
    num_classes = 10
    learning_rate = 0.001 
    # declare model object here
    model = MLP(input_size, num_neurons_hidden, num_classes, learning_rate)
    
    # Finally, evaluate the model
    correct_pred = 0
    total_pred = 0
    for inputs, labels in test_loader:
        x = inputs.view(-1, input_size).numpy()
        y = labels.numpy()
        pred = model.forward(x)  # the model refers to the model that was trained during the raining phase
        predicted_labels = np.argmax(pred, 1)
        correct_pred += np.sum(predicted_labels == y)
        total_pred += len(labels)
    print(f"Test Accuracy: {correct_pred/total_pred}")

if __name__ == "__main__":  # Program entry
    main()  
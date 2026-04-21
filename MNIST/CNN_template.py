import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
# ===================== Utility Functions ===================== #

def relu(x):
    return np.maximum(0, x);

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


# ===================== Data Loading ===================== #
def dataloader(train_dataset, test_dataset, batch_size=64):
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def load_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = torchvision.datasets.MNIST(root="./data/mnist", train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root="./data/mnist", train=False, download=True, transform=transform)
    print("Training samples:", len(train_dataset))
    print("Testing samples:", len(test_dataset))
    return dataloader(train_dataset, test_dataset)

# ===================== CNN Structure ===================== #
class CNN:
    def __init__(self, input_size, num_filters, kernel_size, fc_output_size, lr):
        self.input_size = input_size
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.fc_output_size = fc_output_size
        self.lr = lr
        
        self.out_dim = input_size - kernel_size + 1
        self.flatten_size = num_filters * self.out_dim**2
        
        self.kernel = np.random.randn(num_filters, 1, kernel_size, kernel_size) * 0.01
        
        self.w2 = np.random.randn(fc_output_size, self.flatten_size) * 0.01
        self.b2 = np.zeros((fc_output_size, 1)) * 0.01    

        self.x_input = None
        self.z1 = None
        self.za = None
        self.za2 = None
        self.a1_flat = None

    
    
    def forward(self, x):
        """ Forward propagation """
        batch_size = x.shape[0] // self.input_size
        x = x.reshape(batch_size, 1, self.input_size, self.input_size)
    
        self.x_input = x
        
        batch_size, channel, H, W = x.shape

        z1 = np.zeros((batch_size, self.num_filters, self.out_dim, self.out_dim))

        for b in range(batch_size):
            for f in range(self.num_filters):
                for h in range(self.out_dim):
                    for w in range(self.out_dim):
                        # representing the slice where the kernel is, over each images 
                        slice_matrix = x[b, 0, h: h+self.kernel_size, w: w+self.kernel_size]
                        # sum of the slice * the kernel's weight
                        # [f,0]: kernel is 4d, we specified the filter and channel 
                        z1[b,f,h,w] = np.sum(slice_matrix * self.kernel[f,0])

        self.z1 = z1
        a1 = relu(z1)
        self.a1 = a1
        # (64, 676) map of 64 images and the 676 features of each image
        a1_flat = a1.reshape(batch_size, self.flatten_size) 
        self.a1_flat = a1_flat
        z2 = np.dot(a1_flat, self.w2.T) + self.b2.T
        self.z2 = z2

        outputs = softmax(z2) #y_hat

        return outputs

    def backward(self, y, pred):
        """ Backward propagation """
        # 1. one-hot encode the labels
        batch_size = y.shape[0]
        y_one_hot = np.zeros((batch_size, self.fc_output_size))
        y_one_hot[np.arange(batch_size), y] = 1

        # 2. Calculate softmax cross-entropy loss gradient
        dz2 = pred - y_one_hot
        # 3. Calculate fully connected layer gradient
        dw2 = np.dot(dz2.T, self.a1_flat) / batch_size
        db2 = np.sum(dz2, axis=0, keepdims=True).T / batch_size
        dza_flat = np.dot(dz2, self.w2)
        dza = dza_flat.reshape(batch_size, self.num_filters, self.out_dim, self.out_dim)
        
        # 4. Backpropagate through ReLU
        dz1 = dza * (self.z1 > 0)
        # 5. Calculate convolution kernel gradient
        dw1 = np.zeros((self.num_filters, 1, self.kernel_size, self.kernel_size))
        for b in range(batch_size):
            for f in range(self.num_filters):
                for h in range(self.out_dim):
                    for w in range(self.out_dim):
                        slice_matrix = self.x_input[b, 0, h: h+self.kernel_size, w: w+self.kernel_size]
                        dw1[f,0] += slice_matrix * dz1[b,f,h,w]
        dw1 = dw1 / batch_size
        # 6. Update parameters
        self.kernel -= self.lr * dw1
        self.w2 -= self.lr * dw2
        self.b2 -= self.lr * db2
        
        

    def train(self, x, y):
        # call forward function
        outputs = self.forward(x)
        # calculate loss
        batch_size = outputs.shape[0]
        correct_class_probs = outputs[np.arange(batch_size), y]
        loss = -np.mean(np.log(correct_class_probs + 1e-9))

        # call backward function
        self.backward(y, outputs)
        return loss

# ===================== Training Process ===================== #
def main():
    # First, load data
    train_loader, test_loader = load_data()

    # Second, define hyperparameters
    input_size = 28
    num_epochs = 5
    num_filters = 1
    kernel_size = 3
    fc_output_size = 10
    lr = 0.1
    
    model = CNN(input_size, num_filters, kernel_size, fc_output_size, lr)

    # Then, train the model

    start_time = time.time()
    epoch_losses = []
    for epoch in range(num_epochs):
        total_loss = 0

        for inputs, labels in train_loader:  # define training phase for training model
            x = inputs.view(-1, input_size).numpy()
            y = labels.numpy()

            loss = model.train(x, y)
            total_loss += loss

        avg_epoch_loss = total_loss / len(train_loader)
        epoch_losses.append(avg_epoch_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss}") # print the loss for each epoch
    
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Total training time: {training_time:.2f} seconds")

    eval_start_time = time.time()

    correct_pred = 0
    total_pred = 0
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
    plt.title("CNN Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("cnn_training_loss.png")
    plt.show()

if __name__ == "__main__":
    main()

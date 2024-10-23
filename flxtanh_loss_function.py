import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

a=2
q_values = [0.01, 1, 20]
b=5


# Define the FLXtanh activation function
class FLXtanh(nn.Module):
    def __init__(self, q):
        super(FLXtanh, self).__init__()
        self.q=q
    def forward(self, x):
        tanh = (a*(1-self.q*torch.exp(-2*b*x)))/(1 + self.q*torch.exp(-2*b*x))
        return tanh
    
# Define the neural network
class Net(nn.Module):
    def __init__(self, q):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.activation = FLXtanh(q)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the MNIST dataset
mean, std = (0,), (1,)
train_dataset = datasets.MNIST(root='./data', 
                               train=True, 
                               download=True, 
                               #transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean, std)])
                               transform=transforms.transforms.ToTensor()                               
                              )
train_loader = torch.utils.data.DataLoader(train_dataset, 
                                           batch_size=128, 
                                           shuffle=True)

# Initialize the neural network
for q in q_values:
    model = Net(q)

    # Define the loss function, optimizer, and learning rate
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Train the model
    loss_list = []
    for epoch in range(50):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader, 0):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        loss_list.append(running_loss)
        print('Epoch %d loss: %.3f' % (epoch + 1, running_loss))

    # Plot the loss vs iterations curve
    x_values=[i+1 for i in range(50)]
    y_values=loss_list
    label = f'a={a}, b={b}, q={q}'
    plt.plot(x_values, y_values,label=label)
#plt.title('Loss vs Iterations')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.xticks(np.arange(0, 51, step=10))
plt.yticks(np.arange(0, 1000, step=100))
plt.legend()
plt.grid(True)
plt.show()
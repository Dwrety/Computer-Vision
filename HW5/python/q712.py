import torch 
import torch.nn as nn 
import numpy as np 
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms

num_epochs  = 30
num_classes = 10
batch_size = 100
learning_rate = 0.001
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


train_dataset = torchvision.datasets.MNIST(root='../../data/',
                                           train=True, 
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../../data/',
                                          train=False, 
                                          transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=10000, 
                                          shuffle=False)

# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

model = ConvNet(num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_batch = len(train_loader)
loss_plt = []
acc_plt = []
valid_loss_plt = []
valid_acc_plt = []
for epoch in range(num_epochs):
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            valid_acc_plt.append(correct / total)
            valid_loss_plt.append(loss)
    avg_loss = 0
    avg_acc = 0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        predict = outputs.max(1)[1]
        probs = predict == labels
        acc = torch.numel(probs[probs==1])/len(labels)
        avg_acc += acc
        avg_loss += loss.item()
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Batch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_batch, loss.item()))
    avg_loss /= total_batch
    avg_acc /= total_batch
    loss_plt.append(avg_loss)
    acc_plt.append(avg_acc)


    # torch.cuda.empty_cache()

plt.figure()
plt.plot(np.arange(num_epochs),acc_plt)
plt.plot(np.arange(num_epochs),valid_acc_plt)
plt.title('Accuracy')
plt.legend(['Train Accuracy','Validation Accuracy'])
plt.show()

plt.figure()
plt.plot(np.arange(num_epochs),loss_plt)
plt.plot(np.arange(num_epochs),valid_loss_plt)
plt.title('Loss')
plt.legend(['Train Loss', 'Validation Loss'])
plt.show()


# Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')
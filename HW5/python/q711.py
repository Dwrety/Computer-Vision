import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import numpy as np 
import scipy.io 
import matplotlib.pyplot as plt 


train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
test_data = scipy.io.loadmat('../data/nist36_test.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
test_x, test_y = test_data['test_data'], test_data['test_labels']

num_classes = 36
batch_size = 36
num_epoch = 500
input_size = train_x.shape[1]
hidden_size = 64
lr = 5e-2

train_x = torch.from_numpy(np.asarray(train_x)).float()
train_y = torch.from_numpy(np.where(train_y == 1)[1])

valid_x = torch.from_numpy(np.asarray(valid_x)).float()
valid_y = torch.from_numpy(np.where(valid_y == 1)[1])


train_data_torch = torch.utils.data.TensorDataset(train_x,train_y)
train_loader = torch.utils.data.DataLoader(dataset=train_data_torch, batch_size=batch_size,shuffle=True)
valid_data_torch = torch.utils.data.TensorDataset(valid_x,valid_y)


model = nn.Sequential(nn.Linear(input_size, hidden_size),
						nn.Sigmoid(),
						nn.Linear(hidden_size, num_classes),
						nn.Softmax())
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

valid_loss_plt = []
valid_acc_plt = []
loss_plt = []
acc_plt = []
for e in range(num_epoch):
	avg_loss = 0
	avg_acc = 0 
	train_loader = torch.utils.data.DataLoader(dataset=train_data_torch, batch_size=batch_size,shuffle=True)
	num_batch = len(train_loader)
	for batch_x, batch_y in train_loader:
		probs = model(batch_x)
		loss = criterion(probs,batch_y)
		predict = probs.max(1)[1]
		probs = predict == batch_y
		acc = torch.numel(probs[probs == 1])/batch_y.shape[0]
		avg_loss += loss.item()
		avg_acc += acc
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	avg_loss = avg_loss/num_batch
	avg_acc = avg_acc/num_batch
	loss_plt.append(avg_loss)
	acc_plt.append(avg_acc)

	valid_probs = model(valid_x)
	valid_loss = criterion(valid_probs, valid_y)
	valid_predict = valid_probs.max(1)[1]
	valid_probs = valid_predict == valid_y
	valid_acc = torch.numel(valid_probs[valid_probs == 1])/valid_y.shape[0]
	valid_loss_plt.append(valid_loss)
	valid_acc_plt.append(valid_acc)

	if (e+1)%5 == 0:
		print ('Epoch [{}/{}], Validation Loss: {:.4f}, Validation Accuracy: {:.4f}'.format(e+1, num_epoch, valid_loss.item(), valid_acc))

plt.figure()
plt.plot(np.arange(num_epoch),acc_plt)
plt.plot(np.arange(num_epoch),valid_acc_plt)
plt.title('Accuracy')
plt.legend(['Train Accuracy','Validation Accuracy'])
plt.show()

plt.figure()
plt.plot(np.arange(num_epoch),loss_plt)
plt.plot(np.arange(num_epoch),valid_loss_plt)
plt.title('Loss')
plt.legend(['Train Loss', 'Validation Loss'])
plt.show()




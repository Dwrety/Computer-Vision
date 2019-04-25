import torch 
import torch.nn as nn 
import numpy as np 
import matplotlib.pyplot as plt
import torchvision
import os
import matplotlib.patches
import torchvision.transforms as transforms
import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

from q4 import *

def sort_classes(x):
    from sklearn.cluster import MeanShift
    x = x.reshape(-1,1)
    clf = MeanShift(bandwidth=100)
    clf.fit(x)
    return clf.labels_ 

num_epochs  = 20
num_classes = 47
batch_size = 100
learning_rate = 0.001
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

key_map = {0: 48, 1: 49, 2: 50, 3: 51, 4: 52, 5: 53, 6: 54, 7: 55, 8: 56, 9: 57, 10: 65, 11: 66, 12: 67, 13: 68, 14: 69, 15: 70, 16: 71, 17: 72, 18: 73, 19: 74, 20: 75, 21: 76, 22: 77, 23: 78, 24: 79, 25: 80, 26: 81, 27: 82, 28: 83, 29: 84, 30: 85, 31: 86, 32: 87, 33: 88, 34: 89, 35: 90, 36: 97, 37: 98, 38: 100, 39: 101, 40: 102, 41: 103, 42: 104, 43: 110, 44: 113, 45: 114, 46: 116}

train_dataset = torchvision.datasets.EMNIST(root='../../data/',
                                           split='balanced',
                                           train=True, 
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.EMNIST(root='../../data/',
                                          split='balanced',
                                          train=False, 
                                          transform=transforms.ToTensor())

# # print(train_dataset)
# # print(train_dataset)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)

print(train_dataset.size())


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

model.load_state_dict(torch.load('modelemnist.ckpt'))
model.eval()


# # Train the model
# total_batch = len(train_loader)
# loss_plt = []
# acc_plt = []
# valid_loss_plt = []
# valid_acc_plt = []
# for epoch in range(num_epochs):
# 	avg_loss = 0
# 	avg_acc = 0
# 	for i, (images, labels) in enumerate(train_loader):
# 		images = images.to(device)
# 		labels = labels.to(device)

# 		# Forward pass
# 		outputs = model(images)
# 		loss = criterion(outputs, labels)
# 		predict = outputs.max(1)[1]
# 		probs = predict == labels
# 		acc = torch.numel(probs[probs==1])/len(labels)
# 		avg_acc += acc
# 		avg_loss += loss.item()

# 		# Backward and optimize
# 		optimizer.zero_grad()
# 		loss.backward()
# 		optimizer.step()

# 		if (i+1) % 100 == 0:
# 			print ('Epoch [{}/{}], Batch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_batch, loss.item()))
# 	avg_loss /= total_batch
# 	avg_acc /= total_batch
# 	loss_plt.append(avg_loss)
# 	acc_plt.append(avg_acc)


# # Test the model
# model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
# with torch.no_grad():
#     correct = 0
#     total = 0
#     for images, labels in test_loader:
#         images = images.to(device)
#         labels = labels.to(device)
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

#     print('Test Accuracy of the model on the {} test images: {} %'.format(total, 100 * correct / total))	
# torch.save(model.state_dict(), 'modelemnist.ckpt')


# for img in os.listdir('../images'):
#     im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
#     bboxes, bw = findLetters(im1)
#     bboxes[bboxes < 0] = 0
#     line_of_text = []

    # plt.imshow(bw,cmap='gray')
    # for bbox in bboxes:
    #     minr, minc, maxr, maxc = bbox
    #     rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
    #                             fill=False, edgecolor='red', linewidth=2)
    #     plt.gca().add_patch(rect)
    # plt.show()

    # classes = sort_classes(bboxes[:,2])
    # indexes = np.unique(classes, return_index=True)[1]
    # line = [classes[index] for index in sorted(indexes)]
    # bboxes = np.c_[bboxes,classes].astype(int)
    # for l in line:
    #     bboxes_line = bboxes[bboxes[:,4]==l]
    #     bboxes_line = bboxes_line[bboxes_line[:,1].argsort()]
    #     previous = bboxes_line[0,3]
    #     for box in bboxes_line:
    #         y1,x1,y2,x2 = box[0:4]
    #         image = bw[y1:y2,x1:x2]
    #         image[0:9,:] = 1
    #         image[-10:,:] = 1
    #         image[:,0:9] = 1
    #         image[:,-10:] = 1
    #         image = skimage.transform.resize(image, (28,28),anti_aliasing=False).T
            
    #         # image = np.asarray(image, dtype=np.uint8)
    #         image = 1 - image
    #         # plt.imshow(image,cmap='gray')
    #         # plt.show()
    #         image = image.reshape(1,1,28,28)
    #         x = torch.from_numpy(np.asarray(image)).float()
    #         # x = torch.utils.data.TensorDataset(x)
    #         x = x.to(device)
    #         outputs = model(x)
    #         _, predicted = torch.max(outputs.data, 1)
    #         if x1-previous>=70:
    #             line_of_text.append(' ')
    #         line_of_text.append(chr(key_map[predicted.item()]))
    #         previous = x2.copy()
    #     line_of_text.append('\n')

    # f = open('outconv.txt', 'a') 
    # for item in line_of_text:
    #     f.write("%s" % item)    


import torch
import torchvision
import torchvision.transforms as transforms
from net import LeNet
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
classes = ('plane', 'car', 'bird', 'cat', 
           'deer', 'dog', 'frog', 'horse', 'ship', 'truch')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 图像预处理函数
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5,0.5))])

# 50000张训练图片
trainset = torchvision.datasets.CIFAR10(root = './data', train = True,
                                        download = False, transform = transform)

# num_workers只能在linux系统下使用，windows必须为0
trainloader = DataLoader(trainset, batch_size = 36,shuffle = True, num_workers = 0)

# 10000张训练图片
testset = torchvision.datasets.CIFAR10(root = './data', train = False,
                                        download = False, transform = transform)

testloader = DataLoader(testset, batch_size = 10000,shuffle = False, num_workers = 0)


import matplotlib.pyplot as plt
import numpy as np

test_data_iter = iter(testloader)
test_image, test_label = test_data_iter.next()
test_image = test_image.to(device)
test_label = test_label.to(device)

# functions to show an image
# def imshow(img):
#     img = img / 2 + 0.5     # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()

# # show images
# imshow(torchvision.utils.make_grid(test_image))
# # print labels
# print(' '.join(f'{classes[test_label[j]]:5s}' for j in range(4)))

network = LeNet().to(device)
loss_function = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(network.parameters(), lr=0.001)

for epoch in range(10):
    running_loss = 0.0
    for step, data in enumerate(trainloader, start = 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = network(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if step % 500 == 499:
            with torch.no_grad():
                outputs = network(test_image)      # [batch, 10]
                predict_y = torch.max(outputs,dim=1)[1] # 最大值对应的标签类别
                accuracy = (predict_y == test_label).sum().cpu().item() / test_label.size(0)

                print('[%d, %5d] train_loss: %.3f test_accuracy: %.3f' %
                      (epoch + 1, step + 1, running_loss / 500, accuracy))
                running_loss = 0.0
print('Finished Training')

save_path = './Lenet.pth'
torch.save(network.state_dict(), save_path)

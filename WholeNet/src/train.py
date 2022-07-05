import torch
import torchvision
from torchvision import transforms, datasets
from net import VGG16, AlexNet, ResNet
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import os
import json
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

preprocession = {
    "train": transforms.Compose(
        [
            # RandomResizedCrop(224)为随机裁剪方法，先随机裁剪，然后缩放尺寸维224*224
            # RandomHorizontalFlip()为随机翻转方法
            # 两者均是增强训练集数据，通过人为随机范围裁剪、缩放和旋转等操作
            # 增大训练集数据多样性，提高模型泛化能力
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5,0.5))
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5,0.5))
        ]
    )
}

# 用于找到照片所在位置
data_root = os.path.abspath(os.path.join(os.getcwd()))
image_path = data_root + "/data_set/flower_data"
# train_dataset.class_to_idx = {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
train_dataset = datasets.ImageFolder(root=image_path+"/train",transform=preprocession["train"])

train_num = len(train_dataset)

flower_list = train_dataset.class_to_idx
# 翻转键值对
cla_dict = dict((val, key) for key, val in flower_list.items())

# 使用json格式进行存储
json_str = json.dumps(cla_dict, indent=4)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

# 设置超参数
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers=0)

validate_dataset = datasets.ImageFolder(root=image_path+"/val",transform=preprocession["val"])
val_num = len(validate_dataset)
validate_loader = DataLoader(validate_dataset, batch_size = batch_size, shuffle=False, num_workers=0)


# network = AlexNet(5, init_weights=True).to(device)
# network = VGG16(5, init_weights=True).to(device)
network = ResNet(5, init_weights=True).to(device)
loss_function = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(network.parameters(), lr=0.0002)

# save_path = './model/AlexNet.pth'
# save_path = './model/VGG16.pth'
save_path = './model/Resnet.pth'
best_acc = 0.0

for epoch in range(100):
    # train()和eval()控制dropout方法的开闭
    network.train()
    running_loss = 0.0
    t1 = time.perf_counter()
    for step, data in enumerate(train_loader, start = 0):
        images, labels = data
        optimizer.zero_grad()
        images = images.to(device)
        labels = labels.to(device)
        outputs = network(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

        rate = (step + 1)/len(train_loader)

        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
    print()
    print(time.perf_counter() - t1)
    network.eval()
    acc = 0.0
    with torch.no_grad():
        for data_test in validate_loader:
            test_images, test_labels = data_test
            test_images = test_images.to(device)
            test_labels = test_labels.to(device)
            outputs = network(test_images)      # [batch, 10]
            predict_y = torch.max(outputs,dim=1)[1] # 最大值对应的标签类别
            acc += (predict_y == test_labels).sum().cpu().item()
        accuracy_test = acc/val_num
        if accuracy_test>best_acc:
            best_acc = accuracy_test
            torch.save(network.state_dict(), save_path)
        print('[%d, %5d] train_loss: %.3f test_accuracy: %.3f' %
                (epoch + 1, step + 1, running_loss / step, accuracy_test))
        running_loss = 0.0

print('Finished Training')
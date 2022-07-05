from matplotlib.transforms import Transform
import torch
import torch.nn.functional as F
from net import LeNet
from PIL import Image
import torchvision.transforms as transforms
classes = ('plane', 'car', 'bird', 'cat', 
           'deer', 'dog', 'frog', 'horse', 'ship', 'truch')
transform = transforms.Compose(
    [transforms.Resize((32, 32)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5,0.5))])

network = LeNet()
network.load_state_dict(torch.load('model\Lenet.pth'))

im = Image.open('test/2.jpeg')
im = transform(im)      # [C, H, W]
im = torch.unsqueeze(im, dim=0)   # [N, C, H, W]

with torch.no_grad():
    outputs = network(im)
    probablity = F.softmax(outputs, dim = 1)
    # 返回最大值的索引的数据部分，并转为numpy格式
    predict = torch.max(probablity, dim=1)[1].data.numpy()
    accuracy = torch.max(probablity, dim=1)[0].data.numpy()
print('accuracy: %.3f, class: %s'%(accuracy, classes[int(predict)]))



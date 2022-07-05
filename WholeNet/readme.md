文件命名规范
所有神经网络训练脚本均包含以下组成部分
1、data_set文件夹：存放数据集和数据集分割算法
2、model文件夹：存放训练好的模型
3、src文件夹：存放所有的源代码
src/net.py：存放所有网络结构类
src/predict.py：存放网络的推理程序
src/train.py：存放网络的训练代码

变量命名规范
1、网络实例化：network = LeNet()
2、初始化类方法必须以下划线(_)开头，如：def _initialize_weights(self):
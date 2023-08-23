import torch
from torch import nn
from torch.ao.quantization.stubs import QuantStub
from torch.nn import Sequential, Conv2d, MaxPool2d
from torch.nn.quantized import Quantize


# 搭建神经网络
class cifar10_net(nn.Module):
    def __init__(self, is_quant = False):
        super(cifar10_net, self).__init__()
        # self.net = Sequential(Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2),
        #                       MaxPool2d(kernel_size=2),
        #                       Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
        #                       MaxPool2d(kernel_size=2),
        #                       Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
        #                       MaxPool2d(kernel_size=2),
        #                       nn.Flatten(),
        #                       nn.Linear(in_features=1024, out_features=64),
        #                       nn.Linear(in_features=64, out_features=10)
        #                       )
        # 使用这一种模型结构有明确的名称定义,对量化模型的加载有利,否则可能会出现网络结构不匹配的错误
        self.conv1 = Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2)
        self.maxpool1 = MaxPool2d(kernel_size=2)
        self.conv2 = Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        self.maxpool2 = MaxPool2d(kernel_size=2)
        self.conv3 = Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.maxpool3 = MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_features=1024, out_features=64)
        self.linear2 = nn.Linear(in_features=64, out_features=10)
        self.is_quant = is_quant
        if is_quant:  # 在模型需要训练后静态量化时,插入量化和反量化节点
            self.quant = torch.quantization.QuantStub()
            self.quant_res = None
            self.dequant = torch.quantization.DeQuantStub()
            self.dequant_res = None

        self.conv1_res = None
        self.maxpool1_res = None
        self.conv2_res = None
        self.maxpool2_res = None
        self.conv3_res = None
        self.maxpool3_res = None
        self.flatten_res = None
        self.linear1_res = None
        self.linear2_res = None

    def forward(self, x):
        # y = self.net(x)
        # return y
        if self.is_quant:  # 需要量化时,在输入侧插入量化节点
            x = self.quant(x)
            self.quant_res = x.detach()
        y = self.conv1(x)
        self.conv1_res = y.detach()
        y = self.maxpool1(y)
        self.maxpool1_res = y.detach()
        y = self.conv2(y)
        self.conv2_res = y.detach()
        y = self.maxpool2(y)
        self.maxpool2_res = y.detach()
        y = self.conv3(y)
        self.conv3_res = y.detach()
        y = self.maxpool3(y)
        self.maxpool3_res = y.detach()
        y = self.flatten(y)
        self.flatten_res = y.detach()
        y = self.linear1(y)
        self.linear1_res = y.detach()
        y = self.linear2(y)
        self.linear2_res = y.detach()
        if self.is_quant:  # 需要量化时,在输出侧插入反量化节点
            y = self.dequant(y)
            self.dequant_res = y.detach()
        return y

# 模型正确性的简单测试
if __name__ == "__main__":
    model = cifar10_net()
    input = torch.ones(64, 3, 32, 32)  # batch_size为64, 3通道
    output = model(input)
    print(output.shape)

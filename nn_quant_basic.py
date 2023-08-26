# 模拟量化神经网络中的各种OP(卷积、池化等)，利用量化后的定点数方案
import numpy as np
import torch
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, ReLU

from model import cifar10_net
from nn_quant_export import quant_export

# 运行quant_export函数,获得字典形式保存的量化参数(scale和zero_point)
model_fp32 = cifar10_net(is_quant=True)
zero_point_dict = {}
quant_dict = {}
fix_point_dict = {}
quant_export(network_class_quant=model_fp32, quant_model_dict_path="./model_pth/model_int8.pth", quant_dict=quant_dict,
             zero_point_dict=zero_point_dict,
             fix_point_dict=fix_point_dict)


# 自定义的定点量化网络,与pytorch生成的量化模型不同点在于：
# 1. 舍去了输入输出端的quant/dequant层,本模型的输入要是已经量化后的activation, 输出由于是比较大小,确定分类结果,故不再对其进行dequant操作
# 对activation的量化可以通过本工程img_quant_export.py完成, 实际上是提前将输入feature量化为uint8,FPGA亦是如此
# 2. 所有层都从文件中读取bias、activation、weight, FPGA亦是如此, 故需要提前运行nn_quant_export.py生成相应的数据文件
# 3. 在Pytorch量化模型中, 卷积层、线性层运算结果需要乘以M = (scale_in*scale_w/scale_out), 在Pytorch生成的量化模型中这一数据是float型
# 为了减少硬件开销, 自定义量化模型使用n位定点数表示这一数据, 会有一定精度损失, 考虑到M<1, 故n位定点数都定义为小数部分.
# 4. Pytorch静态量化模型中, 卷积层、linear层bias依旧是float型,
# 本网络中提前对bias进行量化(scale = scale_in*scale_w, zp = 0, dtype=qint32), 并保存到txt中, 自定义网络读取这一bias进行计算
# 综上,自定义网络的计算全部是定点数计算, 与FPGA中保持一致, 且activation和weight的位宽为8, 硬件开销相较float来说更小
class q_cifar10:
    def __init__(self, n, is_get_intermediate=False):
        # 实例化神经网络各层Class,从量化模型导出的txt文件中读取数据
        self.conv1 = conv2d_txt(in_channels=3, out_channels=32, name="conv1", kernel_size=5, padding=2, n=n)
        self.relu1 = relu_q(name="relu1")
        self.maxpool1 = max_pooling_q(kernel_size=2)

        self.conv2 = conv2d_txt(in_channels=32, out_channels=32, name="conv2", kernel_size=5, padding=2, n=n)
        self.relu2 = relu_q(name="relu2")
        self.maxpool2 = max_pooling_q(kernel_size=2)

        self.conv3 = conv2d_txt(in_channels=32, out_channels=64, name="conv3", kernel_size=5, padding=2, n=n)
        self.relu3 = relu_q(name="relu3")
        self.maxpool3 = max_pooling_q(kernel_size=2)

        self.flatten = Flatten()

        self.linear1 = linear_txt(in_features=1024, out_features=64, n=n, name="linear1")
        self.linear2 = linear_txt(in_features=64, out_features=10, n=n, name="linear2")
        self.is_get_intermediate = is_get_intermediate
        if is_get_intermediate:  # 需要保存中间结果属性
            self.conv1_res = None
            self.conv2_res = None
            self.conv3_res = None
            self.maxpool1_res = None
            self.maxpool2_res = None
            self.maxpool3_res = None
            self.relu1_res = None
            self.relu2_res = None
            self.relu3_res = None
            self.flatten_res = None
            self.linear1_res = None
            self.linear2_res = None

    def forward_q(self, x):
        if self.is_get_intermediate:  # 需要保存中间结果属性
            x = self.conv1.Conv2d(x)
            self.conv1_res = x.detach()
            x = self.relu1.Relu(x)
            self.relu1_res = x.detach()
            x = self.maxpool1.Maxpooling(x)
            self.maxpool1_res = x.detach()

            x = self.conv2.Conv2d(x)
            self.conv2_res = x.detach()
            x = self.relu2.Relu(x)
            self.relu2_res = x.detach()
            x = self.maxpool2.Maxpooling(x)
            self.maxpool2_res = x.detach()

            x = self.conv3.Conv2d(x)
            self.conv3_res = x.detach()
            x = self.relu3.Relu(x)
            self.relu3_res = x.detach()
            x = self.maxpool3.Maxpooling(x)
            self.maxpool3_res = x.detach()

            x = self.flatten(x)
            self.flatten_res = x.detach()

            x = self.linear1.Linear(x)
            self.linear1_res = x.detach()
            x = self.linear2.Linear(x)
            self.linear2_res = x.detach()
        else:
            x = self.conv1.Conv2d(x)
            x = self.relu1.Relu(x)
            x = self.maxpool1.Maxpooling(x)
            x = self.conv2.Conv2d(x)
            x = self.relu2.Relu(x)
            x = self.maxpool2.Maxpooling(x)
            x = self.conv3.Conv2d(x)
            x = self.relu3.Relu(x)
            x = self.maxpool3.Maxpooling(x)
            x = self.flatten(x)
            x = self.linear1.Linear(x)
            x = self.linear2.Linear(x)
        return x


# 从txt读取量化后的第n_img张图片
def read_single_img(file_path, n_img, img_channel, img_size_h, img_size_w):
    # 索引特定的文件位置
    idx_start = (n_img - 1) * img_channel * img_size_h * img_size_w
    idx_end = n_img * img_channel * img_size_h * img_size_w
    with open(file_path, "r") as file:
        hex_strings = file.readlines()
    hex_strings = [s.replace("\n", "") for s in hex_strings]  # 去掉列表中字符串的\n
    hex_int = [int(s, 16) for s in hex_strings]
    img = np.array(hex_int[idx_start:idx_end], dtype="uint8")
    return torch.from_numpy(img).reshape((1, img_channel, img_size_h, img_size_w))


# 读取十六进制表达的txt文件,并将其转换为十进制list输出
def txt_hex_to_dec_list(file_path):
    with open(file_path, "r") as file:
        str_list = file.readlines()
    str_list = [s.replace("\n", "") for s in str_list]  # 去掉列表中字符串的\n
    return [int(s, 16) for s in str_list]  # 转换为十进制list并输出


# 从输入的像素十进制字符列表中读取第n张图片,
# 将访问txt文件并转为十进制list的任务放在函数交给txt_hex_to_dec_list()函数完成,减少读取文件开销
def read_img_from_str_list(int_str_list, n, img_channel, img_size_h, img_size_w):
    # 索引特定的文件位置
    idx_start = (n - 1) * img_channel * img_size_h * img_size_w
    idx_end = n * img_channel * img_size_h * img_size_w
    img = np.array(int_str_list[idx_start:idx_end], dtype="uint8")
    return torch.from_numpy(img).reshape((1, img_channel, img_size_h, img_size_w))


# 从txt文件读取量化后的权重, 返回相应大小的tensor
def read_conv_weight(file_path, kernel_size, in_channels, out_channels):
    with open(file_path, 'r') as file:
        hex_strings = file.read().splitlines()

    # 将16进制字符串转换为整数(uint8)
    hex_values = [int(hex_str, 16) for hex_str in hex_strings]
    # 创建numpy数组
    weight = np.array(hex_values, dtype="int8")
    return torch.from_numpy(weight).reshape((out_channels, in_channels,
                                             kernel_size, kernel_size))


def read_bias(file_path):
    with open(file_path, "r") as file:
        hex_string = file.read().splitlines()
    hex_value = [int(s, 16) for s in hex_string]
    hex_np = np.array(hex_value, dtype=np.uint32)  # 直接使用np.int32报错,故先转为uint32,再变为int32
    hex_np = hex_np.astype(np.int32)
    hex_tensor = torch.from_numpy(hex_np)
    return hex_tensor


# 从txt文件读取线性层量化后的权重, 返回相应大小的tensor
def read_linear_weight(file_path, in_channels, out_channels):
    with open(file_path, 'r') as file:
        hex_strings = file.read().splitlines()

    # 将16进制字符串转换为整数(uint8)
    hex_values = [int(hex_str, 16) for hex_str in hex_strings]
    # 创建numpy数组
    weight_conv1 = np.array(hex_values, dtype="int8")
    return torch.from_numpy(weight_conv1).reshape((out_channels, in_channels))


# 定点量化卷积模拟function,会依据name的值自动访问相关的字典获得zero_point、scale和bias
# 要求这些字典的名称和key值符合本程序设计的规范
# n是s1*s2/s3的定点量化小数点数,默认为16
def q_conv(img, w, b, name, in_channels, out_channels, kernel_size, padding, n):
    # 权重和激活分别减去对应的zero_point, 进行2D卷积
    # 注意激活转换完成后要转为(int8)类型,因为有负数的情况
    img_sub_zp = (img - zero_point_dict["{}.in.zero_point".format(name)])
    img_sub_zp = img_sub_zp.to(torch.int8)
    w_sub_zp = w - quant_dict["{}.weight.zero_point".format(name)]
    img_sub_zp = img_sub_zp.to(torch.float)

    conv = Conv2d(in_channels=in_channels, out_channels=out_channels,
                  kernel_size=kernel_size, padding=padding)
    conv.weight.data = w_sub_zp.to(torch.float)  # 设置卷积权重
    conv.bias.data = b.to(torch.float)  # 设置bias
    conv_res = conv(img_sub_zp).to(torch.int32)  # 卷积
    # 对卷积后的结果乘以量化后的(s1*s2/s3)定点数,并舍去小数部分
    scale_conv_res = ((2 ** (-n) * fix_point_dict["{}.fix.scale".format(name)]) * conv_res).to(torch.int8)
    # 加上输出对应的zero_point, 即为量化卷积计算的结果
    return (scale_conv_res + quant_dict["{}.out.zero_point".format(name)]).to(torch.uint8)


# 量化版本的Relu激活函数, 对于给定量化值, 当值小于其量化参数zero_point时,将数据转换为zero_point, 否则数据保持不变
# 相应的zero_point从与name对应的quant_dict字典中获取, name的命名规范为"relux", 其中x为指定阿拉伯数字
# conv+pooling+relu以及conv+relu的组合顺序可以使用该函数,因为在这种情况下relu输出数据的scale和zero_point保持不变
# 只要知道relu并不改变scale和zero_point即可, 即本函数中使用到的zero_point就是来自上层的zero_point, 在rulu零点与conv零点不同时
# 更换zero_point数据来源即可
def q_relu(img, name):
    zero = quant_dict[name.replace("relu", "conv") + ".out.zero_point"]  # 对应卷积层的量化参数zero_point
    img_zero = torch.zeros_like(img) + zero  # 创建一个"零点"矩阵, 此处"零点"指量化后零点
    return torch.where(img < zero, img_zero, img)  # 小于"零点"的值替换为"zero_point"


# 最大池化,量化版本和非量化版本一致
def q_max_pooling(kernel_size, img):
    max_pooling = MaxPool2d(kernel_size=kernel_size)
    max_pool_res = max_pooling(img.to(torch.float))
    return max_pool_res.to(torch.uint8)


# 平铺
def q_flatten(img):
    flatten = Flatten()
    return flatten(img)


# 量化线性层
def q_linear(w, b, img, name, in_features, out_features, n):
    w_sub_zp = w - quant_dict["{}.weight.zero_point".format(name)]
    in_sub_zp = img - zero_point_dict["{}.in.zero_point".format(name)]
    in_sub_zp = in_sub_zp.to(torch.int8)
    linear = Linear(in_features=in_features, out_features=out_features)
    linear.bias.data = b.to(torch.float)
    linear.weight.data = w_sub_zp.to(torch.float)
    out = linear(in_sub_zp.to(torch.float)).to(torch.int32).detach()
    # 乘以对应的量化后的(s1*s2/s3)scale,并保留整数位,加上输出的zero_point即为最终数据
    scale_out = ((2 ** (-n) * fix_point_dict["{}.fix.scale".format(name)]) * out) + quant_dict[
        "{}.out.zero_point".format(name)]
    return scale_out.to(torch.uint8)


class conv2d_txt:
    # 读取txt文件(权重、bias),并进行量化后的卷积,其中n代表(s1*s2/s3)定点量化后的小数位
    # 此class模拟了FPGA中的定点卷积运算
    def __init__(self, name, in_channels, out_channels, kernel_size, n, padding):
        self.name = name
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n = n
        self.padding = padding
        # 加载量化后的权重、激活以及量化后的bias的文件路径
        # 实例化此class时,应先运行nn_quant_export.py导出相关文件等
        self.weight_path = "./txt/{}.weight_int8.txt".format(name)
        self.bias_path = "./txt/{}.bias_int32.txt".format(name)
        # 初始化时暂时没有实际加载数据
        self.b = None
        self.w = None

    def Conv2d(self, img):
        # # 调用自定义的数据加载函数
        self.b = read_bias(self.bias_path)
        self.w = read_conv_weight(file_path=self.weight_path, in_channels=self.in_channels,
                                  out_channels=self.out_channels, kernel_size=self.kernel_size)
        # 调用自定义的量化卷积函数
        return q_conv(name=self.name, w=self.w, b=self.b, in_channels=self.in_channels, out_channels=self.out_channels,
                      n=self.n, padding=self.padding, img=img, kernel_size=self.kernel_size)


class linear_txt:
    def __init__(self, name, in_features, out_features, n):
        self.name = name
        self.in_features = in_features
        self.out_features = out_features
        self.n = n
        self.weight_path = "./txt/{}.weight_int8.txt".format(name)
        self.bias_path = "./txt/{}.bias_int32.txt".format(name)
        # 调用自定义函数从txt读取权重及bias
        self.b = read_bias(self.bias_path)
        self.w = read_linear_weight(self.weight_path, in_channels=in_features, out_channels=out_features)

    def Linear(self, img):
        # 调用自定义量化线性处理函数
        return q_linear(w=self.w, b=self.b, img=img, name=self.name,
                        in_features=self.in_features, out_features=self.out_features, n=self.n)


class max_pooling_q:
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size

    def Maxpooling(self, img):
        # 调用自定义的池化函数
        return q_max_pooling(kernel_size=self.kernel_size, img=img)

class relu_q:
    def __init__(self, name):
        self.name = name
    def Relu(self, x):
        return q_relu(img=x, name=self.name)




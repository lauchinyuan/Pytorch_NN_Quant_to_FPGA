# 将输入特征图依据quantStub的参数进行量化, 并将结果保存到txt
import os.path

import numpy as np
import torch
import torchvision.datasets
from torch.utils.data import DataLoader
from nn_quant_export import quant_export
from model import cifar10_net
# 运行quant_export函数,获得字典形式保存的量化参数(scale)用以量化测试数据集
model_fp32 = cifar10_net(is_quant=True)
quant_dict = {}
quant_export(network_class_quant=model_fp32, quant_model_dict_path="./model_pth/model_int8.pth", quant_dict=quant_dict)

# 导入测试数据
test_data = torchvision.datasets.CIFAR10("./dataset", train=False, download=True,
                                         transform=torchvision.transforms.ToTensor())
loader = DataLoader(test_data, batch_size=1)

# 将图片量化,保存为1列uint8,存到txt中,每次运行开销大,暂时注释掉,需要导出新的image时取消注释
file_path = "./txt/img_uint8.txt"
if os.path.exists(file_path):   # 若文件存在,删除原来的文件
    os.remove(file_path)
with open(file_path, "ab") as f:  # 打开文件并接续写入
    for data in loader:
        img, target = data
        img_q = torch.quantize_per_tensor(img, scale=quant_dict["quant.out.scale"],
                                          zero_point=quant_dict["quant.out.zero_point"], dtype=torch.quint8)
        img_q_int = img_q.int_repr()
        img_q_int = torch.reshape(img_q_int, (-1, 1)).numpy()
        np.savetxt(f, img_q_int, fmt="%02x")

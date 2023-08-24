# 将输入特征图依据quantStub的参数进行量化, 并将结果保存到txt
import os.path

import numpy as np
import torch
import torchvision.datasets
from torch.utils.data import DataLoader


def img_quant_export(quant_dict, save_path: str = "./txt/img_uint8.txt"):
    # quant_dict: 输入的量化参数字典
    # save_path: 量化后文件的保存路径
    # 导入测试数据
    test_data = torchvision.datasets.CIFAR10("./dataset", train=False, download=True,
                                             transform=torchvision.transforms.ToTensor())
    loader = DataLoader(test_data, batch_size=1)

    # 将图片量化,保存为1列uint8,存到txt中,每次运行开销大,暂时注释掉,需要导出新的image时取消注释
    if os.path.exists(save_path):  # 若文件存在,删除原来的文件
        os.remove(save_path)
    with open(save_path, "ab") as f:  # 打开文件并接续写入
        for data in loader:
            img, target = data
            img_q = torch.quantize_per_tensor(img, scale=quant_dict["quant.out.scale"],
                                              zero_point=quant_dict["quant.out.zero_point"], dtype=torch.quint8)
            img_q_int = img_q.int_repr()
            img_q_int = torch.reshape(img_q_int, (-1, 1)).numpy()
            np.savetxt(f, img_q_int, fmt="%02x")

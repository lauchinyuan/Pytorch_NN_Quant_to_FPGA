#  量化后模型参数导出, 以便在FPGA上实现推理
import os.path

import numpy as np
import torch

from model import cifar10_net


def quant_export(network_class_quant, quant_model_dict_path: str,
                 quant_dict={}, scale_dict={}, fix_point_dict={}, zero_point_dict={},
                 txt_path_dir: str = "txt", coe_path_dir: str = "coe"):
    # network_class_quant: 加入量化节点的浮点网络实例
    # quant_model_dict_path: 量化后模型参数字典路径
    # 后续FPGA运算参考模拟程序nn_forward_verilog、输入图片量化程序img_quant_export中会用到的各种参数字典如下:
    # quant_dict: 保存各层量化参数(scale\zero_point)的字典
    # scale_dict: 保存各层输入数据(activation)scale值的字典
    # zero_point_dict: 保存各层输入数据(activation)zero_point值的字典
    # fix_point_dict: 将系数M=(scale_in * scale_weight/scale_out)这一浮点数进行定点量化后的结果, 保存到该字典
    # txt_path_dir: 保存txt文件的文件夹路径名称

    # 创建相关文件夹
    if not os.path.exists("./{}".format(txt_path_dir)):
        os.makedirs("./{}".format(txt_path_dir))
    if not os.path.exists("./{}".format(coe_path_dir)):
        os.makedirs("./{}".format(coe_path_dir))

    # 量化后模型的加载,与pytorch量化处理保持一致
    model_fp32 = network_class_quant
    state_dict = torch.load(quant_model_dict_path)
    model_fp32.qconfig = torch.quantization.get_default_qconfig('x86')
    load_model_prepared = torch.quantization.prepare(model_fp32)
    model_int8 = torch.quantization.convert(load_model_prepared)
    model_int8.load_state_dict(state_dict)

    bias_dict = {}  # 保存各层浮点偏置bias的字典, 对此字典的值进行提前量化, 最后保存为txt文档
    # 量化后模型参数导出
    for key in state_dict:
        # print(key)
        if "bias" in key:
            # 先将bias存在字典中,后续将会进一步结合相关scale进行量化后保存
            bias_dict[key] = state_dict[key].detach()
        if "weight" in key:  # 提取量化后权重到txt文件,文件保存为1列,
            # 保存量化信息(scale、zero_point)到字典
            quant_dict[key + ".scale"] = state_dict[key].q_scale()
            quant_dict[key + ".zero_point"] = state_dict[key].q_zero_point()
            weight_int = state_dict[key].int_repr()
            weight_int = torch.reshape(weight_int, (-1, 1))
            # 转成uint8是因为np.savetxt只支持无符号16进制保存,实际上二进制值并不改变
            weight_int_np = weight_int.numpy().astype("uint8")
            np.savetxt("./{}/{}_int8.txt".format(txt_path_dir, key), weight_int_np, fmt="%02x")

            # 保存为coe文件
            coe_path = "./{}/{}_int8.coe".format(coe_path_dir, key)
            write_np2coe(path=coe_path, data_np=weight_int_np, radix=16, fmt="{:02x}")

        # 获取各层输出的Zero_point & scale, 保存到量化参数字典
        # 注意这里输出的是各层输出结果的量化参数, 而不是权重的量化参数
        if ("scale" in key) or ("zero_point" in key):
            # 替换字符串,相当于在原来的key中间插入.out,方便后续辨识
            key_replace = key.replace(".scale", ".out.scale").replace(".zero_point", ".out.zero_point")
            quant_dict[key_replace] = state_dict[key].item()

        # 某些层(线性层)的模型参数形式是元组
        if ("_packed_params" in key) and ("dtype" not in key):
            w, b = state_dict[key]
            bias_key = key.replace("_packed_params._packed_params", "bias")
            bias_dict[bias_key] = b.detach()

            # 保存权重w对应的scale和zero_point
            quant_dict[key.replace("_packed_params._packed_params", "weight.scale")] = w.q_scale()
            quant_dict[key.replace("_packed_params._packed_params", "weight.zero_point")] = w.q_zero_point()

            # 对权重进行处理, 保存为1列txt
            weight_int = torch.reshape(w.int_repr(), (-1, 1))
            weight_int_np = weight_int.numpy().astype("uint8")
            # print(weight_int_np)
            txt_path = "./{}/".format(txt_path_dir) + key.replace("_packed_params._packed_params", "") + \
                       "weight_int8.txt"
            np.savetxt(txt_path, weight_int_np, fmt="%02x")

            # 保存为coe文件
            coe_path = "./{}/".format(coe_path_dir) + key.replace("_packed_params._packed_params", "") + \
                       "weight_int8.coe"
            write_np2coe(path=coe_path, data_np=weight_int_np, radix=16, fmt="{:02x}")

    # 对bias进行量化处理需要用到两个输入变量的scale相乘(s1*s2),
    # 为了方便使用字典key,将原来使用previous_layer.out.scale作为key的字典重新改为以next_layer.in.scale
    # 这样可以通过next_layer名称索引到scale_dict[next_layer.in.scale]以及scale_dict[next_layer.weight.scale],将两者相乘即为bias量化用到的scale

    # 定义神经网络需要用到的scale的OP顺序列表
    nn_op_order = ["quant", "conv1", "conv2", "conv3", "linear1", "linear2"]

    # scale字典创建更名
    # 实际上实现了以下key的更名：
    # quant.out.scale --> conv1.in.scale
    # conv1.out.scale --> conv2.in.scale
    # -----------------------------etc----
    # previous_layer.out.scale --> next_layer.in.scale
    # zero_point同理

    i = 1
    for op in nn_op_order:
        if op != nn_op_order[-1]:
            scale_dict[nn_op_order[i] + ".in.scale"] = quant_dict[op + ".out.scale"]
            zero_point_dict[nn_op_order[i] + ".in.zero_point"] = quant_dict[op + ".out.zero_point"]
            i = i + 1

    # 依据scale字典对bias(float32)进行量化,并保存至txt文件,以便FPGA调用
    for key in bias_dict:
        b = bias_dict[key]
        s1 = scale_dict[key.replace(".bias", ".in.scale")]  # 来自前一级input的scale
        s2 = quant_dict[key.replace(".bias", ".weight.scale")]  # 自身op权重weight的scale

        # 量化
        b_q = torch.quantize_per_tensor(b, scale=s1 * s2, zero_point=0, dtype=torch.qint32)
        b_q_int = b_q.int_repr().numpy().astype("uint32")

        # 保存到txt
        np.savetxt("./{}/".format(txt_path_dir) + key + "_int32.txt", b_q_int, fmt="%08x")

        # 保存到coe
        coe_path = "./{}/".format(coe_path_dir) + key + "_int32.coe"
        write_np2coe(path=coe_path, data_np=b_q_int, radix=16, fmt="{:08x}")

    # 依据scale字典, 计算[s1(in.scale)*s2(weight.scale)]/s3(out.scale)经过n位定点量化后的值,然后保存到字典
    n = 16
    for key in scale_dict:
        # print(key)
        s1 = scale_dict[key]
        s2 = quant_dict[key.replace("in.scale", "weight.scale")]
        s3 = quant_dict[key.replace("in.scale", "out.scale")]
        M = s1 * s2 / s3
        M_int = int(M * 2**n)  # 定点量化后的值
        fix_point_dict[key.replace(".in", ".fix")] = M_int


# 逐条查看字典值
def view_dict(dict):
    for key in dict:
        print("{}: {}".format(key, dict[key]))


def write_np2coe(path: str, data_np, radix: int = 16, fmt: str = "{:02x}"):
    # 将numpy数据转换为vivado ROM 初始化coe文件

    # path:  欲保存的路径
    # radix: 保存的进制
    # fmt:   格式指定符
    with open(path, "w") as f:
        # coe文件的文件头
        coe_title1 = "MEMORY_INITIALIZATION_RADIX = {};".format(radix)  # 进制
        coe_title2 = "MEMORY_INITIALIZATION_VECTOR = "
        # 写入coe文件头
        f.write(coe_title1 + "\n")
        f.write(coe_title2 + "\n")
        # 写入中间数据, 以","结尾
        for i in range(len(data_np) - 1):
            data = int(data_np[i])
            f.write(("{},\n".format(fmt)).format(data))
        # 写入最后一个数据, 以";"结尾
        data = int(data_np[-1])
        f.write(("{};".format(fmt)).format(data))


if __name__ == "__main__":
    quant_dict_path = "./model_pth/model_int8.pth"
    model_fp32 = cifar10_net(is_quant=True)
    scale_dict = {}
    zero_point_dict = {}
    quant_dict = {}
    fix_point_dict = {}
    quant_export(network_class_quant=model_fp32, quant_model_dict_path=quant_dict_path,
                 quant_dict=quant_dict, scale_dict=scale_dict, zero_point_dict=zero_point_dict,
                 fix_point_dict=fix_point_dict)

    # 查看各层量化参数
    print("_________________________quant_____________________________")
    view_dict(quant_dict)

    # 查看scale字典数据
    print("_________________________scale_____________________________")
    view_dict(scale_dict)

    # 查看fix_point字典数据
    print("_________________________fix_____________________________")
    view_dict(fix_point_dict)

    # zero_point
    print("__________________________zero___________________________")
    view_dict(zero_point_dict)

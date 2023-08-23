#  量化后模型参数导出, 以便在FPGA上实现推理
import numpy as np
import torch

from model import cifar10_net


def quant_export(network_class_quant, quant_model_dict_path: str, bias_dict={},
                 quant_dict={}, scale_dict={}, fix_point_dict={}, zero_point_dict={},
                 txt_path_dir: str = "txt"):
    # 量化后模型的加载
    model_fp32 = network_class_quant
    state_dict = torch.load(quant_model_dict_path)
    model_fp32.qconfig = torch.quantization.get_default_qconfig('x86')
    load_model_prepared = torch.quantization.prepare(model_fp32)
    model_int8 = torch.quantization.convert(load_model_prepared)
    model_int8.load_state_dict(state_dict)

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
            np.savetxt(
                "./{}/".format(txt_path_dir) + key.replace("_packed_params._packed_params", "") + "weight_int8.txt",
                weight_int_np, fmt="%02x")

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

    # 依据scale字典, 计算[s1(in.scale)*s2(weight.scale)]/s3(out.scale)经过n位定点量化后的值,然后保存到字典
    n = 16
    for key in scale_dict:
        # print(key)
        s1 = scale_dict[key]
        s2 = quant_dict[key.replace("in.scale", "weight.scale")]
        s3 = quant_dict[key.replace("in.scale", "out.scale")]
        M = s1 * s2 / s3
        M_int = int(M * 2 ** n)  # 定点量化后的值
        fix_point_dict[key.replace(".in", ".fix")] = M_int


# 逐条查看字典值
def view_dict(dict):
    for key in dict:
        print("{}: {}".format(key, dict[key]))


# quant_dict_path = "./model_pth/model_int8.pth"
# model_fp32 = cifar10_net(is_quant=True)
# scale_dict = {}
# zero_point_dict = {}
# bias_dict = {}
# quant_dict = {}
# fix_point_dict = {}
# quant_export(network_class_quant=model_fp32, quant_model_dict_path=quant_dict_path,
#              bias_dict=bias_dict, quant_dict=quant_dict, scale_dict=scale_dict, zero_point_dict=zero_point_dict,
#              fix_point_dict=fix_point_dict)


# # 查看各层量化参数
# print("_________________________quant_____________________________")
# view_dict(quant_dict)
#
# # 查看bias字典数据
# print("_________________________bias_____________________________")
# view_dict(bias_dict)
#
# # 查看scale字典数据
# print("_________________________scale_____________________________")
# view_dict(scale_dict)
#
# # 查看fix_point字典数据
# print("_________________________fix_____________________________")
# view_dict(fix_point_dict)
#
# print("__________________________zero___________________________")
# view_dict(zero_point_dict)

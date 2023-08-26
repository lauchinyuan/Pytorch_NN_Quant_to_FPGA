# from nn_static_quant import static_quant
# from train import model_train
# from model import cifar10_net
# from nn_quant_export import quant_export
# from img_quant_export import img_quant_export
#
# if __name__ == "__main__":
#     # 由于采用的是训练后静态量化方案,因此训练时无需例化插入量化节点的模型
#     model = cifar10_net(is_quant=False)
#
#     # # 训练模型, 训练完成后保存为pth文件, 无需重新训练时建议注释掉
#     # model_train(network_class=model, out_dict_dir="model_pth")
#
#     # 将模型的输入输出端插入量化、反量化节点, 准备进行量化
#     model_quant = cifar10_net(is_quant=True)
#
#     # 量化模型,即通过Pytorch的训练后静态量化方案, 将float32模型转换为int8模型, 量化完成后保存为pth文件
#     static_quant(is_test=True, in_dict_path="./model_pth/model_dict_300.pth",
#                  out_dict_path="./model_pth/model_int8.pth", network_class=model,
#                  network_class_quant=model_quant)
#     # 模型参数(weight、bias)导出为txt, 并通过字典获取后续需要的量化参数(scale及zero_point)
#     quant_dict = {}
#     quant_export(network_class_quant=model_quant, quant_model_dict_path="./model_pth/model_int8.pth",
#                  quant_dict=quant_dict)
#
#     # 依据quant层的量化参数(out_scale、out_zero_point)对测试数据集的所有图片像素进行量化, 并保存为一个txt文件
#     # 后续FPGA硬件直接使用该文件作为特征输入、同时FPGA运算模拟程序nn_forward_verilog也使用该文件作为输入
#     # 计算开销大, 导出保存完毕后注释
#     # img_quant_export(quant_dict=quant_dict, save_path="./txt/img_uint8.txt")
#
#     # 所有参数导出完毕, 若需在Python程序端模拟FPGA的定点计算过程, 运行nn_forward_verilog.py即可查看各层中间结果







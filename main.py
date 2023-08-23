from nn_static_quant import static_quant
from train import model_train
from model import cifar10_net
from nn_quant_export import quant_export
if __name__ == "__main__":

    # 由于采用的是训练后静态量化方案,因此暂时无需例化插入量化节点的模型
    model = cifar10_net(is_quant=False)

    # # 训练模型, 训练完成后保存为pth文件
    # model_train(network_class=model, out_dict_dir="model_pth")

    # 将模型的输入输出端插入量化、反量化节点, 准备进行量化
    model_quant = cifar10_net(is_quant=True)

    # 量化模型
    static_quant(is_test=False, in_dict_path="./model_pth/model_dict_900.pth",
                 out_dict_path="./model_pth/model_int8.pth", network_class=model,
                 network_class_quant=model_quant)
    # 模型参数导出, 这里没有将quant_export函数参数中的字典传出, 使用默认值, 创建空字典
    quant_export(network_class_quant=model_quant, quant_model_dict_path="./model_pth/model_int8.pth")





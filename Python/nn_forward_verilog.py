# 使用经过定点化后的模型进行相关实验测试
# 作为Verilog/FPGA/ASIC计算结果的参考
# 这一过程亦可在MATLAB中进行,但MATLAB不方便进行量化模型的模拟
import torchvision.datasets
from torch.utils.data import DataLoader

from nn_quant_basic import *


# 比较两个模型结果之间的差距,打印相关信息
def error_calculate(res1, res2, name):
    print("________________________{}_res____________________________".format(name))
    # print("res1:{}\nres2:{}".format(res1, res2))
    # print("error:{}".format((res1 - res2).to(torch.int8)))
    print("error_times:{}".format((res1 != res2).sum()))
    print("average error:{}".format(torch.mean(abs((res1 - res2).to(torch.int8).to(torch.float)))))


# 量化后模型的加载, 需要与量化过程中保持一致的处理
model_fp32 = cifar10_net(is_quant=True)
state_dict = torch.load("./model_pth/model_int8.pth")
model_fp32.qconfig = torch.quantization.get_default_qconfig('x86')
load_model_prepared = torch.quantization.prepare(model_fp32)
model_int8 = torch.quantization.convert(load_model_prepared)
model_int8.load_state_dict(state_dict)
# 浮点模型加载, 作为对照
model_fp32 = torch.load("./model_pth/model_900.pth")

model_int8.eval()
model_fp32.eval()

# CIFAR10测试集
batch_size = 1
test_data = torchvision.datasets.CIFAR10("./dataset", train=False, download=True,
                                         transform=torchvision.transforms.ToTensor())
test_loader = DataLoader(test_data, batch_size=batch_size)

# 单张图片输入测试
# # 对量化模型model_int8,输入测试图片,作为自定义scale量化模型的对照
i = 1
for data in test_loader:
    if i == 769:
        img, target = data
        model_int8(img)
        break
    i = i + 1
n = 16
q_net_out = model_int8.linear2_res.int_repr()  # model_int8第linear2层的输出

# 从txt读取量化后的img作为定点量化模型的输入
img_path = "./txt/img_uint8.txt"  # 文件路径
str_list = txt_hex_to_dec_list(img_path)  # 将激活文件转换为十进制字符列表
# 从txt读取量化后测试数据集第n张图片作为定点量化模型的输入
img = read_img_from_str_list(int_str_list=str_list, n=769, img_channel=3, img_size_h=32, img_size_w=32)

# 实例化定点量化模型,输入img,得出输出结果
# 注意这里的定点数小数位n需要与nn_quant_export.py中的保持一致, 需要修改时需要先修改nn_quant_export.py,并运行生成新的fix_scale字典
model_int8_fix = q_cifar10(n=16, is_get_intermediate=True)
fix_net_out = model_int8_fix.forward_q(img)

# 读取定点量化模型model_int8_fix的中间结果,并与量化模型model_int8的中间结果作为对比, 打印相关误差信息, 不需要时注释掉
# error_calculate(model_int8_fix.conv1_res, model_int8.conv1_res.int_repr(), name="conv1")
# error_calculate(model_int8_fix.maxpool1_res, model_int8.maxpool1_res.int_repr(), name="maxpooling1")
# error_calculate(model_int8_fix.conv2_res, model_int8.conv2_res.int_repr(), name="conv2")
# error_calculate(model_int8_fix.maxpool2_res, model_int8.maxpool2_res.int_repr(), name="maxpooling2")
# error_calculate(model_int8_fix.conv3_res, model_int8.conv3_res.int_repr(), name="conv3")
# error_calculate(model_int8_fix.maxpool3_res, model_int8.maxpool3_res.int_repr(), name="maxpooling3")
# error_calculate(model_int8_fix.flatten_res, model_int8.flatten_res.int_repr(), name="flatten")
# error_calculate(model_int8_fix.linear1_res, model_int8.linear1_res.int_repr(), name="linear1")
# error_calculate(model_int8_fix.linear2_res, model_int8.linear2_res.int_repr(), name="linear2")

# 单张图片测试结果总结
print("model_int8_fix_target:{}".format(fix_net_out.argmax(1)))
print("model_int8_target:{}".format(q_net_out.argmax(1)))
print("model_int8_fix_out:{}".format(fix_net_out))
print("model_int8_out:    {}".format(q_net_out))


# 整个测试数据集进行测试,统计总的正确率
# # 用测试数据集比较量化模型、定点量化模型、浮点模型的预测准确性
i = 1
total_accuracy_int8 = 0  # 预测正确数目统计
total_accuracy_fp32 = 0
total_accuracy_fix = 0
total_test_times = len(test_data)
for data in test_loader:
    img, target = data  # 浮点数据,喂到量化模型model_int8及浮点模型model_fp32
    img_q = read_img_from_str_list(str_list, n=i, img_channel=3, img_size_w=32,
                                   img_size_h=32)  # 定点数据输入到定点量化模型
    int8_out = model_int8(img)  # 量化模型
    fp32_out = model_fp32(img)  # 浮点模型
    fix_out = model_int8_fix.forward_q(img_q)  # 定点量化模型

    # 三类模型预测的结果
    int8_target = int8_out.argmax(1)
    fp32_target = fp32_out.argmax(1)
    fix_target = fix_out.argmax(1)

    # 统计
    accuracy_int8 = (int8_target == target).sum()
    accuracy_fp32 = (fp32_target == target).sum()
    accuracy_fix = (fix_target == target).sum()
    total_accuracy_int8 = total_accuracy_int8 + accuracy_int8
    total_accuracy_fp32 = total_accuracy_fp32 + accuracy_fp32
    total_accuracy_fix = total_accuracy_fix + accuracy_fix
    if i % 1000 == 0:  # 每500轮输出提示信息
        print("第{}轮测试完成".format(i))
        print("已输入{}张测试图片".format(batch_size * i))
        print("浮点模型预测正确数量:{}".format(total_accuracy_fp32))
        print("量化模型预测正确数量:{}".format(total_accuracy_int8))
        print("定点量化模型预测正确数量:{}".format(total_accuracy_fix))
    i = i + 1
print("浮点模型预测正确率:{}".format(total_accuracy_fp32 / total_test_times))
print("量化模型预测正确率:{}".format(total_accuracy_int8 / total_test_times))
print("定点量化模型预测正确率:{}".format(total_accuracy_fix / total_test_times))
print("浮点模型-->定点模型, 预测正确率下降:{}%".format(
    ((total_accuracy_fp32 - total_accuracy_int8) / total_accuracy_fp32) * 100))
print("定点模型-->量化定点模型, 预测正确率下降:{}%".format(
    ((total_accuracy_int8 - total_accuracy_fix) / total_accuracy_int8) * 100))
print("浮点模型-->量化定点模型, 预测正确率下降:{}%".format(
    ((total_accuracy_fp32 - total_accuracy_fix) / total_accuracy_fp32) * 100))




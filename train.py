import os
import time

import torch
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader


def model_train(network_class, out_dict_dir: str = "model_pth", batch_size: int = 64,
                lr: float = 0.001, epoch: int = 20, n: int = 20):
    # out_dict_dir: 输出模型文件保存的文件夹名称
    # lr： 学习率
    # epoch: 训练轮数
    # n: 每n个epoch训练保存一次

    if not os.path.exists("./{}".format(out_dict_dir)):  # 若不存在文件夹则创建
        os.makedirs("./{}".format(out_dict_dir))

    # 准备数据集
    train_data = torchvision.datasets.CIFAR10("./dataset", download=True, train=True,
                                              transform=torchvision.transforms.ToTensor())

    test_data = torchvision.datasets.CIFAR10("./dataset", download=True, train=False,
                                             transform=torchvision.transforms.ToTensor())

    # 加载数据
    train_loader = DataLoader(train_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    # 创建网络模型
    model = network_class

    # 或者导入已经部分训练过的模型
    # model = torch.load("./model_pth/model_900.pth", map_location=torch.device("cpu"))
    if torch.cuda.is_available():
        model = model.cuda()  # 如果可以使用CUDA加速,将模型转移到GPU上

    # 创建损失函数
    loss = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        loss = loss.cuda()

    # 定义优化器(SGD随机梯度下降)
    optim = torch.optim.SGD(model.parameters(), lr=lr)

    # 训练次数记录
    total_train_step = 0
    # 测试次数记录
    total_test_step = 0

    # 添加tensorboard可视化工具
    writer = SummaryWriter("logs")
    start_clk = time.time()  # 程序时间统计,计数开始
    test_data_size = len(test_data)
    for i in range(epoch):
        print("第 {} 轮处理开始".format(i + 1))
        total_loss = 0  # 查看每一轮处理后的损失函数数据
        model.train()  # 调整模型为训练状态
        for train_data in train_loader:  # 获取训练数据
            imgs, targets = train_data  # 得到训练数据的标签和特征
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            out = model(imgs)  # 模型处理
            loss_res = loss(out, targets)  # 损失函数
            total_loss = total_loss + loss_res  # 累计损失
            optim.zero_grad()  # 优化器梯度清零
            loss_res.backward()  # 损失函数反向传播,重新计算新梯度
            optim.step()  # 优化器优化处理
            total_train_step = total_train_step + 1
            if total_train_step % 100 == 0:  # 每逢100输出一次结果
                print("第 {} 次训练,本次训练loss = {}".format(total_train_step, loss_res))
        print("第 {} 轮处理结束, train_loss = {}".format(i + 1, total_loss))
        writer.add_scalar("train_loss", total_loss, total_train_step)

        # 每训练完一轮进行一次测试,评估模型正确性
        total_test_loss = 0
        model.eval()  # 调整模型为验证状态
        with torch.no_grad():
            total_accuracy = 0
            for test_data in test_loader:
                imgs, targets = test_data
                if torch.cuda.is_available():
                    imgs = imgs.cuda()
                    targets = targets.cuda()
                output = model(imgs)
                loss_res_test = loss(output, targets)
                total_test_loss = total_test_loss + loss_res_test.item()
                prd_target = output.argmax(1)
                accuracy = (prd_target == targets).sum()  # 单次测试的正确结果个数
                total_accuracy = total_accuracy + accuracy
        # 结果分析
        print("test_loss = {}".format(total_test_loss))
        print("整体测试集上的正确率为 {}".format(total_accuracy / test_data_size))
        writer.add_scalar("test_loss", total_test_loss, total_test_step)
        writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)
        total_test_step = total_test_step + 1

        # 保存训练好的模型
        if (i+1) % n == 0:
            torch.save(model, "./{}/model_{}.pth".format(out_dict_dir, i+1))  # 保存结构和参数
            torch.save(model.state_dict(), "./{}/model_dict_{}.pth".format(out_dict_dir, i+1))  # 保存成字典

        # 每个epoch统计一次时间, 为了效率可以注释掉
        end_clk = time.time()
        runtime = end_clk - start_clk
        runtime = time.strftime("%H:%M:%S", time.gmtime(runtime))
        print("训练测试已运行时长:{}".format(runtime))

    writer.close()

import argparse # 导入argparse库，用于处理命令行参数，使得脚本可以从命令行接收不同的配置。
import logging # 导入logging库，用于记录日志，便于调试和跟踪程序运行状态。
import os # 导入os库，用于与操作系统进行交互，如创建目录等。
import random # 导入random库，用于生成随机数。
import sys # 导入sys库，用于与Python解释器进行交互，这里用于将日志同时输出到控制台。
import time # 导入time库，这里没有直接使用，但在更复杂的训练循环中常用于计时。
import numpy as np # 导入NumPy库，并将其别名为np，用于处理数值计算和数组操作。
import torch # 导入PyTorch库，用于构建和训练神经网络。
import torch.nn as nn # 导入PyTorch的神经网络模块，nn是所有神经网络模块的基类。
import torch.optim as optim # 导入PyTorch的优化器模块，如SGD, Adam等。
from tensorboardX import SummaryWriter # 导入SummaryWriter，用于将训练过程中的数据写入TensorBoard，便于可视化。
from torch.nn.modules.loss import CrossEntropyLoss # 导入交叉熵损失函数，常用于多类别分类任务。
from torch.utils.data import DataLoader # 导入DataLoader，用于高效地加载数据，支持批量和多线程加载。
from tqdm import tqdm # 导入tqdm库，用于创建进度条，可视化循环进度。
from utils import DiceLoss # 从utils模块导入DiceLoss，常用于医学图像分割任务。
from torchvision import transforms # 导入torchvision的transforms模块，用于图像变换和数据预处理。
from utils import test_single_volume # 从utils模块导入test_single_volume函数，用于评估单个测试卷的性能。
from torch.nn import functional as F # 导入torch.nn.functional，其中包含许多无状态的函数，如激活函数和插值函数。
from datasets.dataset_synapse import Synapse_dataset, RandomGenerator # 从自定义的datasets模块导入Synapse_dataset和RandomGenerator，用于加载和预处理Synapse数据集。

import matplotlib.pyplot as plt # 导入matplotlib库，用于绘制图表。
import pandas as pd # 导入pandas库，用于数据处理和分析，这里用于处理结果数据。
import datetime # 导入datetime库，用于获取当前时间，用于命名保存文件。

# 该函数用于在训练结束后对模型进行推理（推理），评估其在测试集上的性能。
def inference(model, testloader, args, test_save_path=None):
    # model (torch.nn.Module): 待评估的网络模型。
    # testloader (torch.utils.data.DataLoader): 测试数据集加载器。
    # args (argparse.Namespace): 包含所有命令行参数的对象。
    # test_save_path (str, optional): 保存测试结果的路径。
    
    model.eval() # 将模型设置为评估模式。这会关闭dropout和batch normalization等训练时的行为。
    metric_list = 0.0 # 初始化一个变量，用于累积所有样本的评估指标。

    # 使用tqdm创建进度条，遍历testloader中的每个批次。
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        # i_batch (int): 当前批次的索引。
        # sampled_batch (dict): 包含测试样本的字典，通常有'image'、'label'和'case_name'等键。
        
        # 获取图像和标签的高度和宽度。
        h, w = sampled_batch["image"].size()[2:] 
        # 从批次中提取图像、标签和病例名称。由于testloader的batch_size=1，case_name被提取为单个字符串。
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0] 
        # 调用test_single_volume函数对单个样本进行评估。
        # metric_i (list of lists): 返回一个包含dice和hd95分数的列表，每个类别一个子列表。
        metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
        # 将当前样本的指标转换为NumPy数组并累加到总和中。
        metric_list += np.array(metric_i) 
        # 记录当前样本的评估日志，包括病例名称、平均dice和平均hd95分数。
        logging.info(' ==训练结束后对模型进行推理的日志记录==   idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    
    # 计算所有样本的平均指标。
    metric_list = metric_list / len(testloader.dataset)
    
    # 遍历每个类别（从1到num_classes-1），记录每个类别的平均dice和hd95。
    for i in range(1, args.num_classes):
        logging.info('==训练结束后对模型进行推理的日志记录==  Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
    
    # 计算所有类别的平均dice分数。
    performance = np.mean(metric_list, axis=0)[0] 
    # 计算所有类别的平均hd95分数。
    mean_hd95 = np.mean(metric_list, axis=0)[1] 
    
    # 记录最终的平均性能指标。
    logging.info('==训练结束后对模型进行推理的日志记录==Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    
    # 返回最终的平均dice和hd95分数。
    return performance, mean_hd95

# 该函数用于绘制并保存训练结果曲线。
def plot_result(dice, h, snapshot_path,args):
    # dice (list): 包含每个评估周期平均dice分数的列表。
    # h (list): 包含每个评估周期平均hd95分数的列表。
    # snapshot_path (str): 保存结果的目录。
    # args (argparse.Namespace): 命令行参数。
    
    dict = {'mean_dice': dice, 'mean_hd95': h} # 创建一个字典，用于构建DataFrame。
    df = pd.DataFrame(dict) # 使用字典创建pandas DataFrame。
    
    plt.figure(0) # 创建第一个绘图窗口，用于绘制dice曲线。
    df['mean_dice'].plot() # 绘制mean_dice列的曲线。
    resolution_value = 1200 # 设置保存图像的分辨率。
    plt.title('Mean Dice') # 设置图表标题。
    date_and_time = datetime.datetime.now() # 获取当前时间。
    filename = f'{args.model_name}_' + str(date_and_time)+'dice'+'.png' # 构造文件名，包含模型名、时间戳和指标名。
    save_mode_path = os.path.join(snapshot_path, filename) # 拼接保存路径。
    plt.savefig(save_mode_path, format="png", dpi=resolution_value) # 保存图表为PNG文件。
    
    plt.figure(1) # 创建第二个绘图窗口，用于绘制hd95曲线。
    df['mean_hd95'].plot() # 绘制mean_hd95列的曲线。
    plt.title('Mean hd95') # 设置图表标题。
    filename = f'{args.model_name}_' + str(date_and_time)+'hd95'+'.png' # 构造文件名。
    save_mode_path = os.path.join(snapshot_path, filename) # 拼接保存路径。
    plt.savefig(save_mode_path, format="png", dpi=resolution_value) # 保存图表。

    # 保存结果到CSV文件。
    filename = f'{args.model_name}_' + str(date_and_time)+'results'+'.csv' # 构造CSV文件名。
    # 保存训练的模型的权重路径
    save_mode_path = os.path.join(snapshot_path, filename) # 拼接保存路径。
    df.to_csv(save_mode_path, sep='\t') # 将DataFrame保存为CSV文件，使用制表符作为分隔符。


# 这是主训练函数，用于控制整个训练流程，包括数据加载、模型训练、验证和保存。
def trainer_synapse(args, model, snapshot_path):
    # args (argparse.Namespace): 命令行参数。
    # model (torch.nn.Module): 待训练的网络模型。
    # snapshot_path (str): 保存日志和模型权重的目录。
    
    # 创建测试结果保存目录。
    os.makedirs(os.path.join(snapshot_path, 'test'), exist_ok=True) 
    test_save_path = os.path.join(snapshot_path, 'test') # 拼接测试结果的保存路径。

    # 配置日志记录器。日志信息将同时写入文件和控制台。
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args)) # 记录所有命令行参数到日志中。

    base_lr = args.base_lr # 获取基础学习率。
    num_classes = args.num_classes # 获取类别数量。
    batch_size = args.batch_size * args.n_gpu # 计算总批次大小。
    
    # 定义图像预处理transforms。
    x_transforms = transforms.Compose([
        transforms.ToTensor(), # 将PIL图像转换为PyTorch张量，维度从(H, W, C)变为(C, H, W)，值从[0, 255]变为[0, 1]。
        transforms.Normalize([0.5], [0.5]) # 对张量进行标准化，(x - 0.5) / 0.5，将值映射到[-1, 1]。
    ])
    # 定义标签预处理transforms。
    y_transforms = transforms.ToTensor() # 将PIL图像转换为PyTorch张量，维度从(H, W, C)变为(C, H, W)，值从[0, 255]变为[0, 1]。

    # 创建训练数据集实例。
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",img_size=args.img_size,
                               norm_x_transform = x_transforms, norm_y_transform = y_transforms)
    
    # 打印训练集长度。
    print("The length of train set is: {}".format(len(db_train)))
    
    # 定义一个辅助函数，用于为每个DataLoader工作进程设置随机种子，确保数据加载的可复现性。
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    # 创建训练数据加载器。
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    # 创建测试数据集实例。
    db_test = Synapse_dataset(base_dir=args.test_path, split="test_vol", list_dir=args.list_dir, img_size=args.img_size)
    # 创建测试数据加载器。这里batch_size=1，因为测试时通常逐个样本处理。
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

    # 如果有多个GPU，使用nn.DataParallel来并行化模型。
    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    model.train() # 将模型设置为训练模式，激活dropout和batch normalization等层。

    # 定义损失函数和优化器。
    ce_loss = CrossEntropyLoss() # 交叉熵损失。
    dice_loss = DiceLoss(num_classes) # Dice损失。
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001) # SGD优化器，设置学习率、动量和权重衰减。
    writer = SummaryWriter(snapshot_path + '/log') # 创建SummaryWriter实例，用于TensorBoard。
    
    # 初始化训练状态变量。
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader) # 计算总迭代次数。
    
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))

    best_performance = 0.0 # 初始化最佳性能指标。
    iterator = tqdm(range(max_epoch), ncols=70) # 创建epoch循环的进度条。
    dice_=[] # 用于存储每个评估周期的平均dice分数。
    hd95_= [] # 用于存储每个评估周期的平均hd95分数。

    # 开始主训练循环。
    for epoch_num in iterator:
        # 遍历trainloader中的每个批次。
        for i_batch, sampled_batch in enumerate(trainloader):
            # 获取图像批次和标签批次。
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            # 将数据移动到CUDA设备上。label_batch.squeeze(1)是为了去除额外的一维通道维度。
            # image_batch 的形状为 (B, C, H, W)， label_batch 的形状为 (B, H, W)。
            image_batch, label_batch = image_batch.cuda(), label_batch.squeeze(1).cuda() 
            # 前向传播：将图像输入模型得到输出。
            # outputs 的形状为 (B, num_classes, H, W)。
            outputs = model(image_batch)
            
            # 计算交叉熵损失。
            # inputs: outputs (B, num_classes, H, W)
            # targets: label_batch (B, H, W)，类型为long。
            loss_ce = ce_loss(outputs, label_batch[:].long()) 
            # 计算Dice损失。
            # inputs: outputs (B, num_classes, H, W)
            # targets: label_batch (B, H, W)
            # softmax=True: 表示在计算损失前对outputs进行softmax操作。
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            # 综合损失，CE和Dice损失按0.4和0.6的权重进行加权。
            loss = 0.4 * loss_ce + 0.6 * loss_dice 
            
            # 梯度清零，反向传播，优化器步进。
            optimizer.zero_grad() # 清除之前的梯度。
            loss.backward() # 计算梯度。
            optimizer.step() # 更新模型参数。

            # 学习率调度：使用多项式衰减策略。
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9 
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_ # 更新优化器中的学习率。

            iter_num = iter_num + 1 # 迭代次数加1。
            # 记录标量数据到TensorBoard。
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)

            # 记录日志。
            logging.info('iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' % (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))

            # 每隔20次迭代，将训练图像、预测结果和真实标签写入TensorBoard。
            if iter_num % 20 == 0:
                # 图像处理：获取第2个样本（索引1）的第1个通道，并进行Min-Max归一化，以便正确显示。
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                # 预测结果处理：对outputs进行softmax，然后argmax得到类别索引，并进行缩放以便可视化。
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                # 真实标签处理：获取第2个样本的标签，增加维度，并进行缩放以便可视化。
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        # 在每个epoch结束时进行评估。
        eval_interval = args.eval_interval 
        # 当epoch_num达到总epoch数的一半且满足评估间隔时，执行评估。
        if epoch_num >= int(max_epoch / 2) and (epoch_num + 1) % eval_interval == 0:
            # 构造模型保存路径。
            filename = f'{args.model_name}_epoch_{epoch_num}.pth'
            save_mode_path = os.path.join(snapshot_path, filename)
            # 保存模型的权重。
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            
            logging.info("*" * 20)
            logging.info(f"==执行推理== Running Inference after epoch {epoch_num}")
            print(f"Epoch {epoch_num}")
            # 调用inference函数进行评估。
            mean_dice, mean_hd95 = inference(model, testloader, args, test_save_path=test_save_path)
            # 将结果添加到列表中。
            dice_.append(mean_dice)
            hd95_.append(mean_hd95)
            # 将模型重新设置为训练模式。
            model.train()

        # 在最后一个epoch结束时保存模型和评估。
        if epoch_num >= max_epoch - 1:
            filename = f'{args.model_name}_epoch_{epoch_num}.pth'
            save_mode_path = os.path.join(snapshot_path, filename)
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            
            # 如果在最后一个epoch的评估没有被前面的条件触发，则单独执行一次。
            if not (epoch_num + 1) % args.eval_interval == 0:
                logging.info("*" * 20)
                logging.info(f"Running Inference after epoch {epoch_num} (Last Epoch)")
                print(f"Epoch {epoch_num}, Last Epcoh")
                mean_dice, mean_hd95 = inference(model, testloader, args, test_save_path=test_save_path)
                dice_.append(mean_dice)
                hd95_.append(mean_hd95)
                model.train()
                
            iterator.close()
            break # 退出训练循环。
            
    # 调用plot_result函数绘制和保存结果曲线。
    plot_result(dice_, hd95_, snapshot_path, args)
    writer.close() # 关闭SummaryWriter。
    return "Training Finished!" # 返回训练完成信息。
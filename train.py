import argparse # 导入argparse库，用于处理命令行参数，使得脚本可以从命令行接收不同的配置。
import logging # 导入logging库，用于记录日志，便于调试和跟踪程序运行状态。
import os # 导入os库，用于与操作系统进行交互，如创建目录等。
import random # 导入random库，用于生成随机数。
import warnings # 导入warnings库，用于管理警告信息，可以过滤或忽略某些警告。
from pydoc import locate # 导入pydoc中的locate函数，用于根据字符串路径动态地导入和查找模块、类或函数。

import numpy as np # 导入NumPy库，并将其别名为np，用于处理数值计算和数组操作。
import torch # 导入PyTorch库，用于构建和训练神经网络。
import torch.backends.cudnn as cudnn # 导入PyTorch的cudnn库，用于优化卷积神经网络的性能。

from trainer import trainer_synapse # 从名为trainer的模块中导入trainer_synapse函数，这是一个训练主函数。

warnings.filterwarnings("ignore") # 过滤所有警告信息，不让它们在控制台显示，这在大型项目中可以保持输出的整洁。

parser = argparse.ArgumentParser() # 创建一个ArgumentParser对象，它是argparse库的核心，用于解析命令行参数。
# 定义各种命令行参数。每个.add_argument()方法都定义了一个可以从命令行传入的参数，并设置其类型、默认值、帮助信息等。

parser.add_argument( # 定义一个命令行参数。
    "--root_path", # 参数名称。命令行中使用 "--root_path"。
    type=str, # 参数类型为字符串。
    default="data/Synapse/train_npz", # 如果未指定，则使用此默认值。
    help="root dir for train data", # 帮助信息，当用户使用 -h 或 --help 时显示。
)
parser.add_argument(
    "--test_path",
    type=str,
    default="data/Synapse/test_vol_h5",
    help="root dir for test data",
)
parser.add_argument("--dataset", type=str, default="Synapse", help="experiment_name")
parser.add_argument("--list_dir", type=str, default="./lists/lists_Synapse", help="list dir")
parser.add_argument("--num_classes", type=int, default=9, help="output channel of network")
parser.add_argument("--output_dir", type=str, default="./model_out", help="output dir")
parser.add_argument("--max_iterations", type=int, default=90000, help="maximum epoch number to train")
parser.add_argument("--max_epochs", type=int, default=400, help="maximum epoch number to train")
# 实际上batch_size是在训练时候，原作者给出的命令行代码是20.测试的batch_size是24.
parser.add_argument("--batch_size", type=int, default=24, help="batch_size per gpu")
parser.add_argument("--num_workers", type=int, default=4, help="num_workers")
parser.add_argument("--eval_interval", type=int, default=20, help="eval_interval")
parser.add_argument("--model_name", type=str, default="synapse", help="model_name")
parser.add_argument("--n_gpu", type=int, default=1, help="total gpu")
parser.add_argument("--deterministic", type=int, default=1, help="whether to use deterministic training")
parser.add_argument("--base_lr", type=float, default=0.05, help="segmentation network base learning rate")
parser.add_argument("--img_size", type=int, default=224, help="input patch size of network input")
parser.add_argument("--z_spacing", type=int, default=1, help="z_spacing")
parser.add_argument("--seed", type=int, default=1234, help="random seed")
parser.add_argument("--zip", action="store_true", help="use zipped dataset instead of folder dataset")
parser.add_argument(
    "--cache-mode",
    type=str,
    default="part",
    choices=["no", "full", "part"],
    help="no: no cache, "
    "full: cache all data, "
    "part: sharding the dataset into nonoverlapping pieces and only cache one piece",
)
parser.add_argument("--resume", help="resume from checkpoint")
parser.add_argument("--accumulation-steps", type=int, help="gradient accumulation steps")
parser.add_argument(
    "--use-checkpoint", action="store_true", help="whether to use gradient checkpointing to save memory"
)
parser.add_argument(
    "--amp-opt-level",
    type=str,
    default="O1",
    choices=["O0", "O1", "O2"],
    help="mixed precision opt level, if O0, no amp is used",
)
parser.add_argument("--tag", help="tag of experiment")
parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
parser.add_argument("--throughput", action="store_true", help="Test throughput only")
parser.add_argument(
    "--module", help="The module that you want to load as the network, e.g. networks.DAEFormer.DAEFormer"
)

args = parser.parse_args() # 解析所有命令行参数，并将它们作为对象的属性存储在args变量中。

# 确保主程序只在直接运行时执行。
if __name__ == "__main__":
    # 使用pydoc.locate()函数根据args.module参数的字符串路径，动态地导入和定位网络模型类。
    # 例如，如果args.module是"networks.DAEFormer.DAEFormer"，它会加载DAEFormer类。
    transformer = locate(args.module)
    
    # 设置设备为CUDA（如果可用），否则为CPU。这是PyTorch中常用的设备选择方式。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 打印当前使用的设备。
    print("Using device:", device)
    print()

    # 如果设备是CUDA，则打印额外的GPU信息。
    if device.type == "cuda":
        print("==使用的是GPU进行train==")
        print(torch.cuda.get_device_name(0)) # 打印第一个GPU的名称。
        print("Memory Usage:")
        print("Allocated:", round(torch.cuda.memory_allocated(0) / 1024**3, 1), "GB") # 打印已分配的GPU内存。
        print("Cached:   ", round(torch.cuda.memory_reserved(0) / 1024**3, 1), "GB") # 打印缓存的GPU内存。
    
    # 强制设置CUDA可见设备为0号GPU。
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # 根据args.deterministic参数设置PyTorch的cudnn库。
    if not args.deterministic:
        cudnn.benchmark = True # 开启自动寻找最优算法，提高训练速度。
        cudnn.deterministic = False # 允许使用不确定性算法。
    else:
        cudnn.benchmark = False # 关闭自动寻找最优算法。
        cudnn.deterministic = True # 强制使用确定性算法，确保结果可复现。

    # 设置所有随机数生成器的种子，以保证实验结果的可复现性。
    random.seed(args.seed) # 设置Python内置random库的种子。
    np.random.seed(args.seed) # 设置NumPy库的种子。
    torch.manual_seed(args.seed) # 设置PyTorch CPU的种子。
    torch.cuda.manual_seed(args.seed) # 设置PyTorch GPU的种子。

    dataset_name = args.dataset # 将命令行参数中的数据集名称赋值给变量。
    # 定义一个字典，存储不同数据集的配置信息。
    dataset_config = {
        "Synapse": {
            "root_path": args.root_path, # 训练数据的根目录。
            "list_dir": args.list_dir, # 数据列表的目录。
            "num_classes": 9, # 类别数量。
        },
    }

    # 根据批量大小调整学习率，这是一个常见的线性缩放规则。
    if args.batch_size != 24 and args.batch_size % 5 == 0:
        args.base_lr *= args.batch_size / 24
    
    # 更新args中的num_classes、root_path和list_dir，以确保使用dataset_config中的值。
    args.num_classes = dataset_config[dataset_name]["num_classes"]
    args.root_path = dataset_config[dataset_name]["root_path"]
    args.list_dir = dataset_config[dataset_name]["list_dir"]

    # 检查输出目录是否存在，如果不存在则创建它。
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 实例化网络模型。
    # transformer 是 locate() 动态加载的类（例如 DAEFormer）。
    # 使用 args.num_classes 初始化模型。
    # .cuda(0) 将模型移动到第一个GPU上。
    net = transformer(num_classes=args.num_classes).cuda(0)
    print("==网络Network in use:", net.__class__.__name__)# 打印网络名称


    # 定义一个字典，将数据集名称映射到对应的训练器函数。
    trainer = {
        "Synapse": trainer_synapse, # Synapse数据集使用trainer_synapse进行训练。
    }
    
    # 调用对应的训练器函数来开始训练过程。
    # trainer[dataset_name] 会获取到 trainer_synapse 函数。
    # args: 命令行参数对象，包含了所有配置。
    # net: 实例化并已移动到GPU上的网络模型。
    # args.output_dir: 训练输出目录。
    trainer[dataset_name](args, net, args.output_dir)






    # 以下是我的模拟计算 

#     2. 模拟计算与执行
# 我们将以一个具体的命令行输入作为示例，一步步模拟程序的执行。

# 假设命令行输入：

# Bash

# python main.py --module networks.DAEFormer.DAEFormer --dataset Synapse --batch_size 10 --img_size 224
# **注意：**这里 --module 参数是关键，它决定了要加载哪个网络。

# 第一步：参数解析 (args = parser.parse_args())
# argparse 将解析命令行输入，并将参数值赋给 args 对象。

# args.module 被赋值为 'networks.DAEFormer.DAEFormer'。

# args.dataset 被赋值为 'Synapse'。

# args.batch_size 被赋值为 10。

# args.img_size 被赋值为 224。

# 其他未指定的参数将使用其默认值，例如：

# args.root_path = 'data/Synapse/train_npz'

# args.num_classes = 9

# args.seed = 1234

# args.n_gpu = 1

# 第二步：主程序入口 (if __name__ == "__main__":)
# transformer = locate(args.module): locate 函数会查找并返回 networks.DAEFormer 模块下的 DAEFormer 类。此时，transformer 变量指向 DAEFormer 类本身，而不是其实例。

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu"): 假设您的机器有GPU，device 将被设置为 torch.device('cuda')。

# GPU信息打印：如果设备是CUDA，程序会打印您的GPU型号和当前的内存使用情况。

# os.environ["CUDA_VISIBLE_DEVICES"] = "0": 强制程序只使用0号GPU。

# 确定性设置：args.deterministic 默认为 1（True），因此 cudnn.benchmark 将被设置为 False，cudnn.deterministic 被设置为 True，以确保实验结果的可复现性。

# 随机种子设置：

# random.seed(1234)

# np.random.seed(1234)

# torch.manual_seed(1234)

# torch.cuda.manual_seed(1234)

# 所有这些操作都将随机数生成器的种子设置为 1234，以保证每次运行的结果相同。

# 数据集配置更新：

# dataset_name 被设置为 'Synapse'。

# dataset_config 字典被定义，其中 Synapse 的 num_classes 明确为 9。

# if args.batch_size != 24 and args.batch_size % 5 == 0:: 条件满足，因为 args.batch_size 是 10，10 != 24 且 10 % 5 == 0。

# args.base_lr *= 10 / 24：初始学习率 0.05 会被更新为 0.05 * (10 / 24) = 0.02083。

# args.num_classes 被设置为 dataset_config['Synapse']['num_classes']，即 9。

# args.root_path 被设置为 dataset_config['Synapse']['root_path']，即 'data/Synapse/train_npz'。

# args.list_dir 被设置为 dataset_config['Synapse']['list_dir']，即 './lists/lists_Synapse'。

# 创建输出目录：os.path.exists("./model_out") 检查输出目录，如果不存在，os.makedirs("./model_out") 会创建它。

# 模型实例化：

# net = transformer(num_classes=args.num_classes).cuda(0)

# 这行代码等同于 net = DAEFormer(num_classes=9).cuda(0)。

# 一个 DAEFormer 模型实例被创建，其输出通道数设置为 9。

# 模型实例被移动到0号GPU上，准备进行训练。

# 启动训练：

# trainer[dataset_name] 获取 trainer_synapse 函数。

# trainer_synapse(args, net, args.output_dir) 被调用，训练过程正式开始。

# trainer_synapse 函数将使用 args 对象中所有配置（如学习率、批次大小、图像大小等），加载数据，并开始循环训练模型 net，其结果将被保存到 args.output_dir 指定的目录中。
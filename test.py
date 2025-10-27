import argparse # 导入argparse库，用于解析命令行参数，使程序可配置。
import logging # 导入logging库，用于记录程序运行中的信息、警告和错误。
import os # 导入os库，用于与操作系统交互，如文件路径操作。
import random # 导入random库，用于生成伪随机数，确保结果可复现。
import sys # 导入sys库，提供对解释器使用或维护的一些变量的访问。

import numpy as np # 导入NumPy库，用于科学计算和数组操作。
import torch # 导入PyTorch库，用于构建和训练神经网络。
import torch.backends.cudnn as cudnn # 导入cudnn，PyTorch中用于加速卷积运算的库。
import torch.nn as nn # 导入PyTorch的神经网络模块，nn是所有神经网络模块的基类。
from torch.utils.data import DataLoader # 导入DataLoader，用于高效地加载数据集。
from tqdm import tqdm # 导入tqdm，用于在循环中显示进度条。

from datasets.dataset_synapse import Synapse_dataset # 从自定义datasets.dataset模块导入Synapse_dataset类，用于加载Synapse数据集。
from networks.DAEFormer import DAEFormer # 从自定义模块导入DAEFormer模型类。
from trainer import trainer_synapse # 从自定义模块导入训练器函数，虽然此处未用，但被导入。
from utils import test_single_volume # 从自定义模块导入test_single_volume函数，用于对单个3D卷进行测试。

# 创建ArgumentParser对象，用于解析命令行参数。
parser = argparse.ArgumentParser()
# 添加命令行参数，--volume_path，默认值为"/images/PublicDataset/Transunet_synaps/project_TransUNet/data/Synapse/"。
# type=str指定参数类型，help是参数的描述。也就是数据集的路径文件夹
parser.add_argument(
    "--volume_path",
    type=str,
    default="./data/Synapse/",
    help="root dir for validation volume data",
)
# 添加--dataset参数，默认值为"Synapse"。 也就是数据集的名字
parser.add_argument("--dataset", type=str, default="Synapse", help="experiment_name")
# 添加--num_classes参数，默认值为9。
parser.add_argument("--num_classes", type=int, default=9, help="output channel of network")
# 添加--list_dir参数，默认值为"./lists/lists_Synapse"。
parser.add_argument("--list_dir", type=str, default="./lists/lists_Synapse", help="list dir")
# 添加--output_dir参数，默认值为"./model_out"。
parser.add_argument("--output_dir", type=str, default="./model_out", help="output dir")
# 添加--max_iterations参数，默认值为30000。
parser.add_argument("--max_iterations", type=int, default=30000, help="maximum epoch number to train")
# 添加--max_epochs参数，默认值为400。
parser.add_argument("--max_epochs", type=int, default=400, help="maximum epoch number to train")
# 添加--batch_size参数，默认值为24。
parser.add_argument("--batch_size", type=int, default=24, help="batch_size per gpu")
# 添加--img_size参数，默认值为224，表示输入图像的尺寸。
parser.add_argument("--img_size", type=int, default=224, help="input patch size of network input")
# 添加--is_savenii参数，一个布尔标志，如果存在则为True。
parser.add_argument("--is_savenii", action="store_true", help="whether to save results during inference")
# 添加--test_save_dir参数，默认值为"../predictions"。
parser.add_argument("--test_save_dir", type=str, default="../predictions", help="saving prediction as nii!")
# 添加--deterministic参数，默认值为1，控制是否使用确定性算法。
parser.add_argument("--deterministic", type=int, default=1, help="whether use deterministic training")
# 添加--base_lr参数，默认值为0.05，表示学习率。
parser.add_argument("--base_lr", type=float, default=0.05, help="segmentation network learning rate")
# 添加--seed参数，默认值为1234，表示随机种子。
parser.add_argument("--seed", type=int, default=1234, help="random seed")
# 添加其他与训练和评估相关的参数，如--opts, --zip, --cache-mode, --resume等，这些参数在当前脚本中可能未使用，但被保留。
# 该代码段使用argparse库来定义命令行参数。
# 这使得用户可以在运行脚本时，通过命令行来配置程序的行为，例如指定数据集、模型等。

# 添加一个名为"--opts"的命令行参数。
parser.add_argument(
    "--opts", # 参数名称。用户在命令行中输入--opts来指定这个参数。
    help="Modify config options by adding 'KEY VALUE' pairs. ", # 参数的帮助信息，当用户使用-h或--help时显示。
    default=None, # 参数的默认值。如果用户没有在命令行中提供此参数，其值将为None。
    nargs="+", # 参数的“数量”指示符。
    # nargs='+'表示该参数期望一个或多个值。
    # 例如，用户可以输入：--opts KEY1 VALUE1 KEY2 VALUE2
    # 这些值将被解析为一个列表。
)

# 添加一个名为"--zip"的命令行参数。
parser.add_argument(
    "--zip", # 参数名称。
    # action='store_true'是一个特殊的行为。
    # 它表示当命令行中出现--zip时，该参数的值将被设置为True。
    # 如果命令行中没有出现--zip，则该参数的默认值为False。
    # 这种方式常用于布尔标志。
    action="store_true",
    help="use zipped dataset instead of folder dataset", # 参数的帮助信息，说明其作用是使用压缩数据集。
)

# 添加一个名为"--cache-mode"的命令行参数。
parser.add_argument(
    "--cache-mode", # 参数名称。
    type=str, # 参数的数据类型被指定为字符串（string）。
    default="part", # 参数的默认值为"part"。
    # choices参数限制了该参数的有效值。
    # 用户输入的值必须是列表["no", "full", "part"]中的一个，否则argparse会报错。
    choices=["no", "full", "part"],
    help="no: no cache, " # 参数的帮助信息，详细说明了每个可选值的含义。
    "full: cache all data, "
    "part: sharding the dataset into nonoverlapping pieces and only cache one piece",
)

# 添加一个名为"--resume"的命令行参数。
parser.add_argument(
    "--resume", # 参数名称。
    # help参数说明该参数用于从检查点恢复训练。
    # 由于没有指定type或default，它的默认值将是None。
    help="resume from checkpoint",
)

# 添加一个名为"--accumulation-steps"的命令行参数。
parser.add_argument(
    "--accumulation-steps", # 参数名称。
    type=int, # 参数的数据类型为整数（int）。
    # help参数说明该参数用于梯度累积步数。
    help="gradient accumulation steps",
)

# 添加一个名为"--use-checkpoint"的命令行参数。
parser.add_argument(
    "--use-checkpoint", # 参数名称。
    # action='store_true'表示一个布尔标志。
    # 当命令行中出现--use-checkpoint时，该参数值为True，否则为False。
    action="store_true",
    help="whether to use gradient checkpointing to save memory", # 帮助信息，说明其作用是使用梯度检查点以节省内存。
)

# 添加一个名为"--amp-opt-level"的命令行参数。
parser.add_argument(
    "--amp-opt-level", # 参数名称。
    type=str, # 参数数据类型为字符串。
    default="O1", # 参数默认值为"O1"。
    # choices参数限制了该参数的有效值为["O0", "O1", "O2"]。
    choices=["O0", "O1", "O2"],
    help="mixed precision opt level, if O0, no amp is used", # 帮助信息，说明该参数用于混合精度训练的优化级别。
)

# 添加一个名为"--tag"的命令行参数。
parser.add_argument(
    "--tag", # 参数名称。
    # help参数说明该参数用于标记实验。
    help="tag of experiment",
)

# 添加一个名为"--eval"的命令行参数。
parser.add_argument(
    "--eval", # 参数名称。
    # action='store_true'表示一个布尔标志。
    # 当命令行中出现--eval时，该参数值为True，否则为False。
    action="store_true",
    help="Perform evaluation only", # 帮助信息，说明其作用是仅执行评估。
)

# 添加一个名为"--throughput"的命令行参数。
parser.add_argument(
    "--throughput", # 参数名称。
    # action='store_true'表示一个布尔标志。
    # 当命令行中出现--throughput时，该参数值为True，否则为False。
    action="store_true",
    help="Test throughput only", # 帮助信息，说明其作用是仅测试吞吐量。
)

# 解析所有命令行参数，并将其作为对象的属性存储在args中。
args = parser.parse_args()
# 如果数据集是Synapse，则更新volume_path。
if args.dataset == "Synapse":
    args.volume_path = os.path.join(args.volume_path, "test_vol_h5")


# 定义推理函数，用于在测试集上评估模型性能。
def inference(args, model, test_save_path=None):
    # 根据args中的配置初始化数据集。
    db_test = args.Dataset(base_dir=args.volume_path, split="test_vol", img_size=args.img_size, list_dir=args.list_dir)
    # 使用DataLoader创建测试数据加载器，batch_size=1，num_workers=1。
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    # 记录测试迭代次数。len(testloader)就是有多少个batchsize
    logging.info("{} test iterations per epoch".format(len(testloader)))
    # 将模型设置为评估模式。
    model.eval()
    metric_list = 0.0 # 初始化评估指标列表。
    # 使用tqdm库遍历testloader，显示进度条。
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        # 从sampled_batch中获取图像和标签的尺寸。
        h, w = sampled_batch["image"].size()[2:]
        # 获取图像、标签和病例名称。sampled_batch["case_name"]是列表，取第一个元素。
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch["case_name"][0]
        # 调用test_single_volume函数对单个病例进行推理和评估。
        metric_i = test_single_volume(
            image, # 输入图像张量。
            label, # 真实标签张量。
            model, # DAEFormer模型。
            classes=args.num_classes, # 类别数。
            patch_size=[args.img_size, args.img_size], # 输入模型的图像块大小。
            test_save_path=test_save_path, # 保存结果的路径。
            case=case_name, # 病例名称。
            z_spacing=args.z_spacing, # z轴间距。
        )
        # 将当前病例的指标累加到metric_list中。np.array(metric_i)将列表转换为NumPy数组以便进行数学运算。
        metric_list += np.array(metric_i)
        # 记录当前病例的平均Dice和HD95。
        logging.info(
            "==日志文件== idx %d case %s mean_dice %f mean_hd95 %f"
            % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1])
        )
    # 计算所有病例的平均指标。
    metric_list = metric_list / len(db_test)
    # 记录每个类别的平均指标。
    for i in range(1, args.num_classes):
        logging.info("==日志文件== Mean class %d mean_dice %f mean_hd95 %f" % (i, metric_list[i - 1][0], metric_list[i - 1][1]))
    # 计算所有类别的平均Dice。
    performance = np.mean(metric_list, axis=0)[0]
    # 计算所有类别的平均HD95。
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    # 记录最终的整体性能。
    logging.info("==日志文件==Testing performance in best val model: mean_dice : %f mean_hd95 : %f" % (performance, mean_hd95))
    # 返回一个字符串，表示推理完成。
    return "==日志文件== Testing Finished!"


# 当脚本作为主程序运行时执行的代码块。
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" # 设置CUDA可见设备为GPU 0。
    if not args.deterministic: # 如果不是确定性训练。
        cudnn.benchmark = True # 开启cuDNN的自动寻找最适合当前硬件的算法，以加速训练。
        cudnn.deterministic = False # 禁用确定性模式。
        print("==不是确定性训练==")
    else: # 如果是确定性训练。
        cudnn.benchmark = False # 关闭cuDNN自动寻找功能。
        cudnn.deterministic = True # 启用确定性模式，牺牲部分速度以确保每次运行结果完全相同。
        print("==确定性训练==")
    random.seed(args.seed) # 设置Python内置random库的随机种子。
    np.random.seed(args.seed) # 设置NumPy的随机种子。
    torch.manual_seed(args.seed) # 设置CPU上的随机种子。
    torch.cuda.manual_seed(args.seed) # 设置GPU上的随机种子。

    # 定义数据集配置字典。
    dataset_config = {
        "Synapse": {
            "Dataset": Synapse_dataset, # 指定数据集类。
            "z_spacing": 1, # 指定z轴间距。
        },
    }
    dataset_name = args.dataset # 从命令行参数获取数据集名称。也就是synapse
    # 从配置字典中获取数据集类和z_spacing，并赋值给args对象。
    args.Dataset = dataset_config[dataset_name]["Dataset"]# 自定义datasets.dataset模块导入Synapse_dataset方法
    args.z_spacing = dataset_config[dataset_name]["z_spacing"]
    args.is_pretrain = True # 设置is_pretrain标志，此处未使用。

    # 初始化DAEFormer模型并将其移动到GPU 0上。
    net = DAEFormer(num_classes=args.num_classes).cuda(0)

    # 确定模型快照的路径。
    snapshot = os.path.join(args.output_dir, "best_model.pth") #output_dir是model_out
    # 如果"best_model.pth"不存在，则尝试加载最后一个epoch的模型。也就是第399个epoch的模型。
    if not os.path.exists(snapshot):
        snapshot = snapshot.replace("best_model", "transfilm_epoch_" + str(args.max_epochs - 1))
    # 加载预训练的模型权重。
    msg = net.load_state_dict(torch.load(snapshot))
    # 打印加载状态信息。
    print("self trained swin unet", msg)
    # 获取快照文件的名称。
    snapshot_name = snapshot.split("/")[-1]

    # 配置日志记录。
    log_folder = "./test_log/test_log_"
    os.makedirs(log_folder, exist_ok=True) # 创建日志文件夹。
    logging.basicConfig( # 配置日志文件的基本设置。
        filename=log_folder + "/" + snapshot_name + ".txt", # 日志文件名。
        level=logging.INFO, # 记录所有INFO及以上级别的信息。
        format="[%(asctime)s.%(msecs)03d] %(message)s", # 日志格式。
        datefmt="%H:%M:%S", # 日期时间格式。
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout)) # 将日志同时输出到控制台。
    logging.info(str(args)) # 记录所有命令行参数。
    logging.info(snapshot_name) # 记录模型快照名称。

    # 根据参数决定是否保存推理结果。
    if args.is_savenii:
        args.test_save_dir = os.path.join(args.output_dir, "predictions")
        test_save_path = args.test_save_dir
        print("==保存推理结果==")
        os.makedirs(test_save_path, exist_ok=True)
    else:
        print("==不保存推理结果==")
        test_save_path = None
    # 调用推理函数，开始评估。
    inference(args, net, test_save_path)
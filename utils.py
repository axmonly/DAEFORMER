import numpy as np # 导入NumPy库，用于数值计算和数组操作。
import torch # 导入PyTorch库，用于构建和训练神经网络。
from medpy import metric # 导入medpy库的metric模块，用于计算医学图像评估指标。
from scipy.ndimage import zoom # 从scipy库导入zoom函数，用于图像的缩放和插值。
import torch.nn as nn # 导入PyTorch的神经网络模块，所有神经网络模块的基类。
import SimpleITK as sitk # 导入SimpleITK库，用于读取、写入和处理医学图像文件，如.nii.gz。
from torch.nn import functional as F # 导入torch.nn.functional，其中包含许多无状态的函数，如激活函数和插值函数。
from torchvision import transforms # 导入torchvision的transforms模块，用于图像变换和数据预处理。


class DiceLoss(nn.Module):
    # n_classes (int): 类别总数，包括背景类。
    def __init__(self, n_classes):
        # 调用父类nn.Module的构造函数，进行必要的初始化。
        super(DiceLoss, self).__init__()
        # 将类别数保存为实例变量，供其他方法使用。
        self.n_classes = n_classes

    # 内部辅助方法，用于将输入的离散类别标签张量转换为独热编码（one-hot encoding）张量。
    def _one_hot_encoder(self, input_tensor):
        # input_tensor (torch.Tensor): 输入的标签张量，形状通常为 (B, H, W)，其中B为批次大小。
        tensor_list = [] # 初始化一个空列表，用于存储每个类别的独热编码张量。
        for i in range(self.n_classes): # 遍历所有类别（从0到n_classes-1）。
            # 对输入张量进行逐像素比较，创建一个布尔张量。
            # 例如，对于类别 i，所有值为 i 的像素位置将变为 True。
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            # 使用unsqueeze(1)方法在维度1（通道维度）上增加一个维度。
            # 这使得temp_prob的形状从(B, H, W)变为(B, 1, H, W)。
            tensor_list.append(temp_prob.unsqueeze(1)) 
        # 使用torch.cat沿着维度1（通道维度）将所有类别的独热编码张量拼接起来。
        # 拼接后的output_tensor形状为 (B, n_classes, H, W)。
        output_tensor = torch.cat(tensor_list, dim=1) 
        # 将张量的数据类型转换为浮点型（float），便于后续数学运算。
        return output_tensor.float()

    # 内部辅助方法，用于计算单个类别（二值）的Dice损失。
    def _dice_loss(self, score, target):
        # score (torch.Tensor): 模型的预测分数，形状为 (B, H, W)，表示单个类别的概率。
        # target (torch.Tensor): 真实标签的独热编码，形状为 (B, H, W)。
        target = target.float() # 确保目标张量的数据类型为浮点型。
        smooth = 1e-5 # 一个小的平滑因子，用于避免分母为零的情况，提高数值稳定性。
        # 计算预测和目标的交集（intersection），即元素级相乘后求和。
        intersect = torch.sum(score * target) 
        # 计算目标的平方和。
        y_sum = torch.sum(target * target) 
        # 计算预测的平方和。
        z_sum = torch.sum(score * score) 
        # 根据Dice系数的公式计算Dice值。
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        # Dice损失通常定义为 1 - Dice系数，表示不重叠的程度。
        loss = 1 - loss
        return loss

    # 前向传播方法，用于计算整个批次的Dice损失。
    def forward(self, inputs, target, weight=None, softmax=False):
        # inputs (torch.Tensor): 模型的原始输出，形状为 (B, n_classes, H, W)。
        # target (torch.Tensor): 真实标签，形状为 (B, H, W)。
        # weight (list, optional): 各个类别的权重，用于处理类别不平衡。
        # softmax (bool): 是否在计算损失前对输入进行softmax操作。
        
        if softmax: # 如果softmax参数为True。
            # 对模型输出进行softmax操作，将值转换为概率分布。
            # inputs的形状保持不变：(B, n_classes, H, W)，但值现在在[0, 1]范围内，且沿着维度1求和为1。
            inputs = torch.softmax(inputs, dim=1)
        # 调用内部方法，将输入的整数标签转换为独热编码。
        target = self._one_hot_encoder(target)
        if weight is None: # 如果没有提供权重。
            # 则将所有类别的权重设置为1。
            weight = [1] * self.n_classes
        # 使用断言（assert）来检查输入和独热编码后的目标张量形状是否匹配，以确保运算正确。
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        
        class_wise_dice = [] # 初始化列表用于存储每个类别的Dice系数。
        loss = 0.0 # 初始化总损失。
        # 遍历每个类别（从0到n_classes-1）。
        for i in range(0, self.n_classes):
            # 使用切片操作inputs[:, i]和target[:, i]来获取对应类别的张量。
            # 调用内部方法_dice_loss，计算单个类别的Dice损失。
            dice = self._dice_loss(inputs[:, i], target[:, i])
            # 将Dice系数（1-损失）添加到列表中，以便返回或记录。
            class_wise_dice.append(1.0 - dice.item())
            # 将当前类别的损失乘以其权重并累加到总损失中。
            loss += dice * weight[i]
        # 返回所有类别损失的平均值。
        return loss / self.n_classes


#### `calculate_metric_percase` 函数
# 该函数用于计算单个二值（前景/背景）分割结果的评估指标（Dice和HD95）。
def calculate_metric_percase(pred, gt):
    # pred (np.ndarray): 模型的预测结果（二值），形状为(H, W)或(D, H, W)。
    # gt (np.ndarray): 真实标签（二值），形状为(H, W)或(D, H, W)。
    
    # 将预测结果中所有大于0的值设置为1，实现二值化。
    pred[pred > 0] = 1 
    # 将真实标签中所有大于0的值设置为1，实现二值化。
    gt[gt > 0] = 1 
    # 检查预测结果和真实标签中是否都包含前景像素。
    if pred.sum() > 0 and gt.sum()>0:
        # 如果都有前景，则计算Dice系数和95% Hausdorff距离。
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95 # 返回这两个指标。
    # 如果预测有前景但真实标签没有（模型误报）。
    elif pred.sum() > 0 and gt.sum()==0:
        # 根据代码逻辑，返回Dice为1，HD95为0。这通常表示一个特定的评估策略，可能意味着没有目标可供预测时，预测有前景是一种错误。
        return 1, 0 
    else:
        # 如果预测和真实标签都没有前景（都是背景），通常视为预测正确，返回0。
        return 0, 0


def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    # image (torch.Tensor): 输入图像张量，形状通常为(1, C, D, H, W)或(1, C, H, W)。
    # label (torch.Tensor): 真实标签张量，形状通常为(1, D, H, W)或(1, H, W)。
    # net (torch.nn.Module): 待评估的网络模型。
    # classes (int): 类别总数。
    # patch_size (list): 模型输入期望的图像块大小，例如[256, 256]。
    # test_save_path (str, optional): 保存结果的路径。
    # case (str, optional): 病例名称，用于命名保存的文件。
    # z_spacing (int): z轴上的间距。

    # 将张量从GPU移到CPU，使用.detach()从计算图中分离，并转换为NumPy数组。
    # .squeeze(0)去除批次维度。如果image初始形状为(1, 1, D, H, W)，转换后为(1, D, H, W)。
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    
    # 检查图像是否为3D（即形状有3个维度）。
    if len(image.shape) == 3:
        # 创建一个与label形状相同的全零数组，用于存储预测结果。
        prediction = np.zeros_like(label) 
        # 遍历3D图像的每个切片（ind代表z轴索引）。
        for ind in range(image.shape[0]):
            slice = image[ind, :, :] # 提取当前切片，形状为 (H, W)。
            x, y = slice.shape[0], slice.shape[1] # 获取切片的尺寸。
            
            # 如果切片尺寸与模型输入期望的patch_size不匹配。
            if x != patch_size[0] or y != patch_size[1]:
                # 使用zoom函数将切片缩放到patch_size，order=3表示使用三次样条插值，适用于连续值图像。
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)
            
            # 定义一个Transforms序列，用于将切片转换为模型所需的张量格式。
            x_transforms = transforms.Compose([
                transforms.ToTensor(), # 将切片转换为张量，形状变为 (1, H, W)。
                transforms.Normalize([0.5], [0.5]) # 标准化，将像素值从[0, 1]映射到[-1, 1]。
            ])
            # 应用transforms，并使用.unsqueeze(0)在第0维增加一个批次维度，.float()确保数据类型为浮点型，.cuda()移到GPU。
            # input的形状为 (1, 1, H', W')，其中H', W'为patch_size。
            input = x_transforms(slice).unsqueeze(0).float().cuda()
            
            net.eval() # 将模型设置为评估模式。
            with torch.no_grad(): # 在此块内不计算梯度，以节省内存和计算。
                outputs = net(input) # 前向传播，得到模型输出。
                # outputs的形状为 (1, num_classes, H', W')。
                # 对输出进行softmax，然后使用argmax得到每个像素的预测类别索引。
                # out的形状为 (H', W')。
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy() # 将预测结果从GPU移到CPU并转换为NumPy数组。
                
                # 如果切片被缩放过。
                if x != patch_size[0] or y != patch_size[1]:
                    # 将预测结果使用zoom函数缩放回原始尺寸。order=0表示最近邻插值，用于保留离散的类别值。
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out # 如果没有缩放，直接使用结果。
                prediction[ind] = pred # 将预测结果存入3D预测数组。
    
    # 处理2D图像的情况。
    else:
        # 将NumPy数组转换为PyTorch张量，并增加两个维度：批次维度和通道维度。
        # input的形状为(1, 1, H, W)。
        input = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            # 前向传播，得到预测类别。
            # out的形状为 (H, W)。
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy() # 转换为NumPy数组。
    
    # 评估指标计算
    metric_list = []
    # 遍历每个类别（从1到classes-1，背景类0通常不评估）。
    for i in range(1, classes):
        # 对预测结果和真实标签进行二值化，并计算该类别的dice和hd95。
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    # 如果提供了保存路径，则使用SimpleITK保存结果。
    if test_save_path is not None:
        # 将NumPy数组转换为SimpleITK图像对象。
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        # 设置图像的物理间距。
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        # 保存为.nii.gz文件。
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    # 返回所有类别的指标列表。
    return metric_list








    # 注解
#     # -------------------- test_single_volume 函数模拟 --------------------
# # 函数入口，假设的输入张量形状：
# image_tensor = torch.randn(1, 1, 64, 512, 512)
# label_tensor = torch.randint(0, 9, (1, 64, 512, 512))

# # 1. 张量转 NumPy 数组
# # image.squeeze(0) 操作后，张量形状为 (1, 64, 512, 512)
# # .cpu().detach().numpy() 后，image_np 形状为 (1, 64, 512, 512)
# image_np = image_tensor.squeeze(0).cpu().detach().numpy()
# # label_np 形状为 (64, 512, 512)
# label_np = label_tensor.squeeze(0).cpu().detach().numpy()

# # 2. 判断图像维度
# # len(image_np.shape) 为 4，所以以下 if len(image.shape) == 3 条件不满足，
# # 但为了模拟代码，我们假设传入的image_tensor为(1, 64, 512, 512)
# # 这样image_np.shape为(64, 512, 512)，len为3，满足条件。

# # 3. 初始化预测数组
# prediction = np.zeros_like(label_np) # prediction 的形状为 (64, 512, 512)

# # 4. 循环切片处理 (假设 ind = 0)
# ind = 0
# slice = image_np[ind, :, :] # 提取第一个切片，slice 的形状为 (512, 512)
# x, y = slice.shape[0], slice.shape[1] # x = 512, y = 512
# patch_size = [224, 224]

# # 5. 尺寸检查与缩放
# if x != patch_size[0] or y != patch_size[1]: # 512 != 224，条件成立
#     # slice 被缩放，使用三次样条插值，得到新的slice_scaled
#     slice_scaled = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)
#     # slice_scaled 的形状变为 (224, 224)

# # 6. 切片预处理
# x_transforms = transforms.Compose([
#     transforms.ToTensor(), # 将 (224, 224) 变为 (1, 224, 224)
#     transforms.Normalize([0.5], [0.5]) # 值从 [0, 1] 变为 [-1, 1]
# ])
# # input 张量形状变为 (1, 1, 224, 224) 并移动到 CUDA
# input_tensor = x_transforms(slice_scaled).unsqueeze(0).float().cuda()

# # 7. 模型推理
# # 假设网络是 U-Net 类型的，输入 (1, 1, 224, 224)
# # outputs_tensor 的形状为 (1, 9, 224, 224)
# outputs_tensor = net(input_tensor)

# # 8. 后处理
# # torch.softmax(...) 形状保持 (1, 9, 224, 224)
# # torch.argmax(...) 沿 dim=1，形状变为 (1, 224, 224)
# # .squeeze(0) 后，out_tensor 的形状为 (224, 224)
# out_tensor = torch.argmax(torch.softmax(outputs_tensor, dim=1), dim=1).squeeze(0)
# # out_np 形状为 (224, 224)，值是整数类别索引 (0-8)
# out_np = out_tensor.cpu().detach().numpy()

# # 9. 结果缩放回原始尺寸
# if x != patch_size[0] or y != patch_size[1]: # 512 != 224，条件成立
#     # pred 使用最近邻插值，将 out_np 从 (224, 224) 缩放到 (512, 512)
#     # pred 的形状为 (512, 512)，值仍然是整数类别
#     pred = zoom(out_np, (x / patch_size[0], y / patch_size[1]), order=0)
# else:
#     pred = out_np

# # 10. 存储预测结果
# # prediction 数组的第 ind (0) 个切片被填充
# prediction[ind] = pred # prediction[0, :, :] 的形状为 (512, 512)

# # 11. 循环结束，prediction 数组被完全填充，形状为 (64, 512, 512)

# # 12. 指标计算
# metric_list = []
# for i in range(1, 9): # 遍历类别 1 到 8
#     # 假设的预测和标签，例如：
#     pred_class_i = (prediction == i) # 形状 (64, 512, 512) 的布尔数组
#     gt_class_i = (label_np == i) # 形状 (64, 512, 512) 的布尔数组
#     # 调用 calculate_metric_percase
#     dice, hd95 = calculate_metric_percase(pred_class_i, gt_class_i)
#     metric_list.append((dice, hd95))
# # metric_list 包含 8 个元组，每个元组代表一个类别的 (dice, hd95)

# # 13. 结果保存
# # 假设 test_save_path 已提供
# # 将 image_np, prediction, label_np 转换为 SimpleITK 图像对象
# # 并以 .nii.gz 格式保存到指定路径

# # 14. 返回
# return metric_list # 返回包含所有类别指标的列表